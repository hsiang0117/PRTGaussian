"""Read-only Cloud-GS RGB data adapter for PRTGaussian.

No CUDA, Torch, renderer, or PLY package is imported at module import time.
The supplied train/test JSON files are authoritative: no split is generated.
Targets are the stored PNG sRGB pixel values, with no per-image normalization.

The geometry-stage target at each training camera is formed from *only* its
available training illuminations: decode sRGB, average linear RGB, encode sRGB.
It is rounded to uint8, like a generated PNG, for the bounded CPU image cache.
This finite training-light average is not claimed to be exact uniform lighting.
The light SH encoder is supplied by the caller; its native basis is not changed.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
from PIL import Image


class PointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True, eq=False)
class CloudFrame:
    split: str
    index: int
    file_path: str
    image_path: Path
    camera_to_world: np.ndarray
    fov_x: float
    width: int
    height: int
    camera_index: int
    time_index: int
    sun_direction: np.ndarray

    @property
    def name(self):
        return str(Path(self.file_path).with_suffix('')).replace('\\', '/').replace('/', '_')

    @property
    def light_id(self):
        return self.time_index

    @property
    def uid(self):
        return self.index

    @property
    def image_name(self):
        return self.name

    @property
    def key(self):
        return self.name

    @property
    def sun_index(self):
        return self.time_index


@dataclass(frozen=True, eq=False)
class MeanFrame:
    """A camera and its training-only illumination members, not a new split."""

    index: int
    members: tuple[CloudFrame, ...]
    split: str = 'train_mean'

    @property
    def camera_index(self):
        return self.members[0].camera_index

    @property
    def camera_to_world(self):
        return self.members[0].camera_to_world

    @property
    def fov_x(self):
        return self.members[0].fov_x

    @property
    def width(self):
        return self.members[0].width

    @property
    def height(self):
        return self.members[0].height

    @property
    def name(self):
        return f'cam{self.camera_index:02d}_train_linear_mean'

    @property
    def image_name(self):
        return self.name

    @property
    def key(self):
        return self.name

    @property
    def sun_index(self):
        return None

    @property
    def uid(self):
        return self.index

    @property
    def light_id(self):
        return None

    @property
    def sun_direction(self):
        return None


def srgb_to_linear(image):
    image = np.asarray(image, dtype=np.float32)
    return np.where(image <= 0.04045, image / 12.92,
                    ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(image):
    image = np.maximum(np.asarray(image, dtype=np.float32), 0.)
    return np.where(image <= 0.0031308, 12.92 * image,
                    1.055 * image ** (1. / 2.4) - 0.055).astype(np.float32)


def native_light_angles(frame_or_direction):
    """Match native my_utils.math_utils.compute_angles (radians, phi then theta).

    `sun_direction` is used in dataset world coordinates. No UE axis conversion
    or sign flip is repeated. Native get_sh_coeffs applies its own phi += pi.
    """
    vector = getattr(frame_or_direction, 'sun_direction', frame_or_direction)
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError('a finite nonzero world-space sun direction is required')
    vector = vector / np.linalg.norm(vector)
    return np.array([np.arctan2(vector[1], vector[0]),
                     np.arccos(np.clip(vector[2], -1., 1.))], dtype=np.float64)


def light_coefficients(frame_or_direction, encoder, order=9):
    """Call unmodified native get_sh_coeffs, returning its original tensor type."""
    if order < 1:
        raise ValueError('SH order must be positive')
    coefficients = encoder(direction=native_light_angles(frame_or_direction), order=order)
    if tuple(coefficients.shape) != (3, order * order):
        raise ValueError(f'native encoder returned {coefficients.shape}, expected (3, {order * order})')
    values = coefficients.detach().cpu().numpy() if hasattr(coefficients, 'detach') else np.asarray(coefficients)
    if not np.isfinite(values).all():
        raise ValueError('native light SH contains non-finite coefficients')
    return coefficients


def camera_matrices(frame, znear=0.01, zfar=100.0):
    """NumPy equivalent of native camera conventions; outputs are transposed.

    OpenGL camera-to-world columns Y/Z are negated to obtain COLMAP axes.
    Projection and transposition follow PRTGaussian's utils/graphics_utils.py
    and scene/cameras.py (Graphdeco/Inria research/evaluation license).
    """
    if not 0 < znear < zfar:
        raise ValueError('require 0 < znear < zfar')
    c2w = frame.camera_to_world.copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    fovy = 2 * math.atan(math.tan(frame.fov_x / 2) * frame.height / frame.width)
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 1 / math.tan(frame.fov_x / 2)
    projection[1, 1] = 1 / math.tan(fovy / 2)
    projection[3, 2] = 1.
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    view = np.asarray(w2c.T, dtype=np.float32)
    projection = projection.T.copy()
    return dict(R=np.asarray(w2c[:3, :3].T, dtype=np.float32),
                T=np.asarray(w2c[:3, 3], dtype=np.float32), FoVx=frame.fov_x,
                FoVy=fovy, znear=znear, zfar=zfar,
                world_view_transform=view, projection_matrix=projection,
                full_proj_transform=view @ projection,
                camera_center=np.asarray(c2w[:3, 3], dtype=np.float32))


class CloudDataset:
    """Metadata, a bounded uint8 LRU, and an optional training-stage target cache."""

    def __init__(self, root, resolution=0, cache_images=16, *, point_cloud_path=None,
                 normalize_scene=False, normalization_margin=0.05):
        self.root = Path(root).expanduser().resolve()
        if int(resolution) != resolution or resolution < 0 or int(cache_images) != cache_images or cache_images < 0:
            raise ValueError('resolution and cache_images must be nonnegative integers')
        self.resolution, self.cache_images = int(resolution), int(cache_images)
        self._images = OrderedDict()
        self._target_cache = {}
        self._target_cache_stage = None
        self._derived_mean_paths = {}
        self.train = self._read_split('train', 0)
        self.test = self._read_split('test', len(self.train))
        overlap = {f.image_path for f in self.train} & {f.image_path for f in self.test}
        if overlap:
            raise ValueError(f'train/test images overlap: {next(iter(overlap))}')
        self.camera_poses = {}
        self.sun_directions = {}
        for frame in self.train + self.test:
            old_pose = self.camera_poses.setdefault(frame.camera_index, frame.camera_to_world)
            old_sun = self.sun_directions.setdefault(frame.time_index, frame.sun_direction)
            if not np.allclose(old_pose, frame.camera_to_world, atol=1e-5, rtol=0):
                raise ValueError(f'camera_index {frame.camera_index} has inconsistent poses')
            if not np.allclose(old_sun, frame.sun_direction, atol=1e-6, rtol=0):
                raise ValueError(f'time_index {frame.time_index} has inconsistent sun directions')
        groups = defaultdict(list)
        for frame in self.train:
            groups[frame.camera_index].append(frame)
        self.stage1 = []
        for i, camera_index in enumerate(sorted(groups)):
            members = groups[camera_index]
            if any((f.width, f.height, f.fov_x) != (members[0].width, members[0].height, members[0].fov_x)
                   for f in members):
                raise ValueError(f'camera {camera_index} has inconsistent intrinsics')
            self.stage1.append(MeanFrame(i, tuple(members)))
        self.point_cloud_path = (Path(point_cloud_path).expanduser().resolve()
                                 if point_cloud_path else self.root / 'points3d.ply')
        if not self.point_cloud_path.is_file():
            raise FileNotFoundError(f'initial PLY must exist; it will not be generated: {self.point_cloud_path}')
        self.normalize_scene = bool(normalize_scene)
        self.center, self.scale, self.offset = np.zeros(3), 1.0, np.zeros(3)
        self.original_bbox = None
        if self.normalize_scene:
            if not 0 <= normalization_margin < .5:
                raise ValueError('normalization_margin must be in [0, 0.5)')
            cloud = self.read_point_cloud(normalized=False)
            lower, upper = cloud.points.min(axis=0), cloud.points.max(axis=0)
            extent = float((upper - lower).max())
            if not np.isfinite(extent) or extent <= 0:
                raise ValueError('PLY bounding box must have positive extent for scene normalization')
            self.original_bbox = [lower.tolist(), upper.tolist()]
            self.center = (lower.astype(np.float64) + upper.astype(np.float64)) / 2.
            self.scale = (1 - 2 * normalization_margin) / extent
            self.offset = np.full(3, .5, dtype=np.float64)

    def _read_split(self, split, uid_offset):
        path = self.root / f'transforms_{split}.json'
        with path.open(encoding='utf-8-sig') as stream:
            metadata = json.load(stream)
        frames, seen, seen_pairs = [], set(), set()
        for i, data in enumerate(metadata['frames']):
            relative = str(data['file_path']).replace('\\', '/')
            image_path = self.root / relative
            if not image_path.suffix:
                image_path = image_path.with_suffix('.png')
            image_path = image_path.resolve()
            if not image_path.is_relative_to(self.root):
                raise ValueError(f'image is outside dataset root: {image_path}')
            if image_path in seen:
                raise ValueError(f'{path}: duplicate image {image_path}')
            seen.add(image_path)
            if image_path.suffix.lower() != '.png':
                raise ValueError(f'expected PNG supervision, got {image_path}')
            with Image.open(image_path) as image:
                width, height = image.size
                if image.mode not in ('RGB', 'RGBA'):
                    raise ValueError(f'{image_path}: expected 8-bit RGB or RGBA, got {image.mode}')
            if self.resolution:
                ratio = min(1., self.resolution / max(width, height))
                width, height = max(1, round(width * ratio)), max(1, round(height * ratio))
            fov = float(data.get('camera_angle_x', metadata.get('camera_angle_x', float('nan'))))
            if not np.isfinite(fov) or not 0 < fov < np.pi:
                raise ValueError(f'{path}, frame {i}: invalid camera_angle_x')
            matrix = np.asarray(data['transform_matrix'], dtype=np.float64)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)):
                raise ValueError(f'{path}, frame {i}: invalid affine transform_matrix')
            rotation = matrix[:3, :3]
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                    or np.linalg.det(rotation) < 0):
                raise ValueError(f'{path}, frame {i}: expected OpenGL rigid camera pose')
            matrix.setflags(write=False)
            camera_index = data.get('camera_index')
            if camera_index is None:
                match = re.search(r'(?:^|/)cam(\d+)(?:/|$)', relative)
                if match is None:
                    raise ValueError(f'{path}, frame {i}: missing camera_index')
                camera_index = int(match.group(1))
            time_index = data.get('time_index', data.get('light_idx'))
            if time_index is None:
                time_index = int(image_path.stem)
            camera_index, time_index = int(camera_index), int(time_index)
            if (camera_index, time_index) in seen_pairs:
                raise ValueError(f'{path}: duplicate camera/sun pair {(camera_index, time_index)}')
            seen_pairs.add((camera_index, time_index))
            direction = np.asarray(data.get('sun_direction'), dtype=np.float64)
            if (direction.shape != (3,) or not np.isfinite(direction).all()
                    or np.linalg.norm(direction) < 1e-10):
                raise ValueError(f'{path}, frame {i}: valid sun_direction is required')
            direction = direction / np.linalg.norm(direction)
            direction.setflags(write=False)
            frames.append(CloudFrame(split, uid_offset + i,
                                     image_path.relative_to(self.root).as_posix(), image_path,
                                     matrix, fov, width, height, camera_index, time_index, direction))
        if not frames:
            raise ValueError(f'{path}: empty split')
        return frames

    def image_uint8(self, frame):
        """Return a read-only HWC RGB array. All retained images use one CPU LRU."""
        key = ('mean', frame.camera_index) if isinstance(frame, MeanFrame) else ('png', frame.image_path)
        if key in self._images:
            value = self._images.pop(key)
            self._images[key] = value
            return value
        if isinstance(frame, MeanFrame) and frame.camera_index in self._derived_mean_paths:
            with Image.open(self._derived_mean_paths[frame.camera_index]) as source:
                value = np.array(source.convert('RGB'), dtype=np.uint8, copy=True)
        elif isinstance(frame, MeanFrame):
            if any(f.split != 'train' for f in frame.members):
                raise ValueError('Stage1 mean must contain training frames only')
            linear_sum = np.zeros((frame.height, frame.width, 3), dtype=np.float64)
            for member in frame.members:
                linear_sum += srgb_to_linear(self.image_uint8(member).astype(np.float32) / 255.)
            rgb = linear_to_srgb(linear_sum / len(frame.members))
            value = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            with Image.open(frame.image_path) as source:
                if source.mode == 'RGBA':
                    rgba = np.asarray(source, dtype=np.uint8)
                    rgb = np.rint(rgba[..., :3].astype(np.float32) *
                                  (rgba[..., 3:4].astype(np.float32) / 255.)).astype(np.uint8)
                    source = Image.fromarray(rgb, mode='RGB')
                else:
                    source = source.convert('RGB')
                if source.size != (frame.width, frame.height):
                    # Same PIL bilinear resizing as the common 3DGS loaders.
                    source = source.resize((frame.width, frame.height), Image.Resampling.BILINEAR)
                value = np.array(source, dtype=np.uint8, copy=True)
        value.setflags(write=False)
        if self.cache_images:
            self._images[key] = value
            while len(self._images) > self.cache_images:
                self._images.popitem(last=False)
        return value

    def image(self, frame):
        """Float32 HWC pixel-space target, without exposure/max normalization."""
        return self.image_uint8(frame).astype(np.float32) / 255.

    @staticmethod
    def _target_key(frame):
        return ('mean', frame.camera_index) if isinstance(frame, MeanFrame) else ('png', frame.image_path)

    def target_cpu(self, frame):
        """Return the original float32 CHW target, optionally reused from CPU RAM.

        Consumers must treat targets as read-only. No CUDA tensor is retained by
        the dataset, and uncached frames (including test frames) are never added
        implicitly. The conversion expression is exactly the previous camera
        loader, so caching changes no alpha, resize, mean, or rounding operation.
        """
        import torch
        key = self._target_key(frame)
        if key in self._target_cache:
            return self._target_cache[key]
        return torch.from_numpy(self.image(frame).transpose(2, 0, 1).copy())

    def prepare_target_cache(self, stage, *, max_bytes=24 * 1024**3):
        """Predecode one training stage outside the loop, without drawing RNG.

        The byte budget is checked from authoritative resized image dimensions
        before any allocation. A stage switch releases the previous targets
        before constructing the next cache. Failure clears partial targets, so
        a retry cannot silently retain a second full cache. Test images cannot
        be requested through this interface. Source images remain read-only.
        """
        if stage not in ('stage1', 'stage2'):
            raise ValueError("target cache stage must be 'stage1' or 'stage2'")
        if isinstance(max_bytes, bool) or int(max_bytes) != max_bytes or max_bytes < 0:
            raise ValueError('max_bytes must be a nonnegative integer')
        frames = self.stage1 if stage == 'stage1' else self.train
        expected_bytes = sum(frame.width * frame.height * 3 * 4 for frame in frames)
        if expected_bytes > max_bytes:
            raise MemoryError(f'{stage} targets require {expected_bytes} bytes; cache budget is {max_bytes}')
        started = time.perf_counter()
        self.clear_target_cache()
        self.clear_cache()
        try:
            for frame in frames:
                self._target_cache[self._target_key(frame)] = self.target_cpu(frame)
            self._target_cache_stage = stage
        except BaseException:
            self.clear_target_cache()
            raise
        finally:
            # The float32 cache fully replaces training-target uint8 reuse.
            self.clear_cache()
        return dict(stage=stage, frames=len(frames), bytes=self.target_cache_bytes,
                    budget_bytes=int(max_bytes), preparation_seconds=time.perf_counter() - started,
                    dtype='float32', layout='CHW', device='cpu', source='training_only',
                    conversion='existing uint8 target -> float32 / 255 -> CHW contiguous')

    @property
    def target_cache_bytes(self):
        return sum(value.numel() * value.element_size() for value in self._target_cache.values())

    def clear_target_cache(self):
        """Release the current stage targets; held camera references remain valid."""
        self._target_cache.clear()
        self._target_cache_stage = None

    def prepare_stage1_means(self, directory):
        """Compute once outside training; keep derived files out of source data."""
        directory = Path(directory).resolve()
        if directory == self.root or directory.is_relative_to(self.root):
            raise ValueError('derived means must not be written inside source data')
        directory.mkdir(parents=True, exist_ok=True)
        for frame in self.stage1:
            path = directory / (frame.name + '.png')
            if path.exists():
                raise FileExistsError(path)
            Image.fromarray(self.image_uint8(frame)).save(path)
            self._derived_mean_paths[frame.camera_index] = path
        self.clear_cache()

    def light_coefficients(self, frame, encoder, order=9):
        return light_coefficients(frame, encoder, order=order)

    @property
    def cache_bytes(self):
        return sum(image.nbytes for image in self._images.values())

    def clear_cache(self):
        self._images.clear()

    def normalize_points(self, points):
        """Apply the one scene similarity to PLY positions or camera centers."""
        points = np.asarray(points)
        return ((points - self.center) * self.scale + self.offset).astype(np.float32)

    def normalized_camera_to_world(self, frame):
        pose = frame.camera_to_world.copy()
        pose[:3, 3] = self.normalize_points(pose[:3, 3])
        return pose

    def camera_matrices(self, frame, znear=0.01, zfar=100.0):
        # Clip distances undergo the same similarity, preserving clipping too.
        proxy = SimpleNamespace(camera_to_world=self.normalized_camera_to_world(frame),
                                fov_x=frame.fov_x, width=frame.width, height=frame.height)
        return camera_matrices(proxy, znear=znear*self.scale, zfar=zfar*self.scale)

    def camera(self, frame, device='cuda', include_image=True, *, znear=0.01, zfar=100.0):
        """Create only this camera's tensors; the dataset retains no GPU objects."""
        import torch
        values = self.camera_matrices(frame, znear=znear, zfar=zfar)
        camera = SimpleNamespace(uid=frame.uid, colmap_id=frame.camera_index,
                                 cam_id=frame.camera_index, image_name=frame.name,
                                 light_id=frame.light_id, sun_direction=frame.sun_direction,
                                 light=frame.sun_direction, image_width=frame.width,
                                 image_height=frame.height, data_device=torch.device(device),
                                 **values)
        for key in ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
            setattr(camera, key, torch.tensor(values[key], dtype=torch.float32, device=device))
        if include_image:
            camera.original_image = self.target_cpu(frame).to(device)
        return camera

    def read_point_cloud(self, normalized=True):
        """Read the existing PLY once; never generate, overwrite, or normalize it."""
        from plyfile import PlyData
        vertices = PlyData.read(str(self.point_cloud_path))['vertex']
        names = vertices.data.dtype.names
        points = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float32)
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError('initial point cloud is empty or non-finite')
        colors = (np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]).astype(np.float32) / 255.
                  if all(k in names for k in ('red', 'green', 'blue')) else np.full_like(points, 0.5))
        normals = (np.column_stack([vertices[k] for k in ('nx', 'ny', 'nz')]).astype(np.float32)
                   if all(k in names for k in ('nx', 'ny', 'nz')) else np.zeros_like(points))
        if not np.isfinite(colors).all() or not np.isfinite(normals).all():
            raise ValueError('initial PLY colors/normals are non-finite')
        if normalized and self.normalize_scene:
            points = self.normalize_points(points)
        return PointCloud(points, colors, normals)

    def normalization(self):
        """Native NeRF++ camera extent, using each training camera once."""
        centers = self.normalize_points(np.stack([f.camera_to_world[:3, 3] for f in self.stage1]))
        center = centers.mean(axis=0)
        radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
        return {'translate': -center, 'radius': radius}

    def manifest(self):
        train_suns = {f.time_index for f in self.train}
        return dict(root=str(self.root), point_cloud_path=str(self.point_cloud_path),
                    train_count=len(self.train), test_count=len(self.test),
                    camera_count=len(self.camera_poses), sun_count=len(self.sun_directions),
                    train_sun_count=len(train_suns), stage1_camera_count=len(self.stage1),
                    test_unseen_sun_count=sum(f.time_index not in train_suns for f in self.test),
                    test_seen_sun_new_combination_count=sum(f.time_index in train_suns for f in self.test),
                    resolution=self.resolution, cache_images=self.cache_images,
                    scene_similarity=dict(enabled=self.normalize_scene, original_bbox=self.original_bbox,
                                          center=self.center.tolist(), scale=self.scale, offset=self.offset.tolist(),
                                          formula='p_normalized = (p_original - center) * scale + offset',
                                          camera_translation_transformed=True, rotations_and_sun_unchanged=True),
                    supervision='stored PNG sRGB / 255; no per-image normalization',
                    stage1_average='training-only sRGB decode -> mean linear RGB -> sRGB -> uint8; no missing-pair synthesis',
                    camera_convention='OpenGL c2w -> flip columns Y/Z -> native transposed COLMAP w2c',
                    train_files=[f.file_path for f in self.train], test_files=[f.file_path for f in self.test],
                    stage1_members={str(f.camera_index): [m.file_path for m in f.members] for f in self.stage1})


class TrainingFrameSampler:
    """One shuffled visit per epoch, restricted to train or training means."""

    def __init__(self, frames, seed=0):
        if not frames or any(frame.split not in ('train', 'train_mean') for frame in frames):
            raise ValueError('sampler requires nonempty training frames')
        self.frames, self.rng, self._pending = list(frames), np.random.default_rng(seed), []

    def next(self):
        if not self._pending:
            self._pending = self.rng.permutation(len(self.frames)).tolist()
        return self.frames[self._pending.pop()]
