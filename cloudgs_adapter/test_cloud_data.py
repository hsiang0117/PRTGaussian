"""CPU-only adapter regression checks. Optional: --dataset D:/dataset/..."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

import cloud_data as data


def fixture(root):
    frames = []
    for camera, light, value in [(0, 0, 0), (0, 1, 255), (1, 0, 100), (0, 2, 255), (1, 1, 200)]:
        path = root / f'cam{camera:02d}/images/{light:04d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 8, 3), value, dtype=np.uint8)).save(path)
        pose = np.eye(4)
        pose[:3, 3] = [camera + 1, 2, 3]
        frames.append(dict(file_path=path.relative_to(root).as_posix(), transform_matrix=pose.tolist(),
                           camera_index=camera, time_index=light, sun_direction=[[1,0,0],[0,1,0],[0,0,1]][light]))
    for split, values in [('train', frames[:3]), ('test', frames[3:])]:
        (root / f'transforms_{split}.json').write_text(json.dumps(dict(camera_angle_x=np.pi/2, frames=values)))
    (root / 'points3d.ply').write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n-2 -1 -3\n2 1 3\n')


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu_fixture_', dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.ds = data.CloudDataset(self.root, cache_images=2)

    def tearDown(self):
        assert self.root.resolve().is_relative_to(Path(__file__).resolve().parent)
        self.temp.cleanup()

    def test_exact_split_and_missing_combinations(self):
        self.assertEqual((len(self.ds.train), len(self.ds.test), len(self.ds.stage1)), (3, 2, 2))
        self.assertEqual([len(x.members) for x in self.ds.stage1], [2, 1])
        self.assertTrue(all(x.split == 'train' for m in self.ds.stage1 for x in m.members))
        self.assertEqual(self.ds.manifest()['test_unseen_sun_count'], 1)
        self.assertEqual(self.ds.manifest()['test_seen_sun_new_combination_count'], 1)
        with self.assertRaises(ValueError):
            data.TrainingFrameSampler(self.ds.test)
        sampler = data.TrainingFrameSampler(self.ds.train, seed=42)
        self.assertEqual({sampler.next().uid for _ in self.ds.train}, {f.uid for f in self.ds.train})

    def test_linear_mean_excludes_test_and_keeps_brightness(self):
        # Mean black and white in linear space is sRGB ~0.735, not 0.5.
        mean = self.ds.image_uint8(self.ds.stage1[0])
        self.assertEqual(int(mean[0, 0, 0]), 188)
        self.assertEqual(int(self.ds.image_uint8(self.ds.stage1[1])[0, 0, 0]), 100)
        self.assertAlmostEqual(float(self.ds.image(self.ds.train[2])[0,0,0]), 100/255, places=6)

    def test_bounded_cpu_cache_readonly_and_source_unchanged(self):
        sources = list(self.root.rglob('*.png')) + [self.root/'points3d.ply']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for frame in self.ds.train + self.ds.test + self.ds.stage1:
            value = self.ds.image_uint8(frame)
            self.assertEqual(value.dtype, np.uint8)
            self.assertFalse(value.flags.writeable)
            self.assertLessEqual(len(self.ds._images), 2)
            self.assertLessEqual(self.ds.cache_bytes, 2*4*8*3)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        self.ds.clear_cache()
        self.assertEqual(self.ds.cache_bytes, 0)

    def test_camera_opengl_colmap_projection(self):
        f = self.ds.train[0]
        m = data.camera_matrices(f)
        np.testing.assert_allclose(m['camera_center'], [1,2,3])
        np.testing.assert_allclose(np.linalg.inv(m['world_view_transform']).T[:3,:3], np.diag([1,-1,-1]))
        # The camera forward point maps to positive z, then center of screen.
        center_forward = np.r_[f.camera_to_world[:3,3] - 5*f.camera_to_world[:3,2], 1]
        view = center_forward @ m['world_view_transform']
        self.assertAlmostEqual(view[2], 5)
        clip = center_forward @ m['full_proj_transform']
        np.testing.assert_allclose(clip[:2]/clip[3], [0,0], atol=1e-6)
        # Positive OpenGL-up maps to negative screen y, as native 3DGS expects.
        up = center_forward.copy(); up[:3] += f.camera_to_world[:3,1]
        self.assertLess((up @ m['full_proj_transform'])[1], 0)
        self.assertAlmostEqual(m['FoVy'], 2*np.arctan(0.5))

    def test_precomputed_training_means_equal_live_means(self):
        expected = [self.ds.image_uint8(f).copy() for f in self.ds.stage1]
        with tempfile.TemporaryDirectory(prefix='derived_', dir=Path(__file__).resolve().parent) as directory:
            self.ds.prepare_stage1_means(directory)
            self.assertEqual(len(list(Path(directory).glob('*.png'))), len(self.ds.stage1))
            for frame, value in zip(self.ds.stage1, expected):
                np.testing.assert_array_equal(self.ds.image_uint8(frame), value)
        with self.assertRaises(ValueError):
            self.ds.prepare_stage1_means(self.root / 'invalid_derived')

    def test_sh_delegate_native_angles_without_basis_changes(self):
        calls = []
        def encoder(direction, order):
            calls.append((direction.copy(), order))
            return np.ones((3, order*order), np.float32)
        result = data.light_coefficients(self.ds.train[1], encoder)
        self.assertEqual(result.shape, (3, 81))
        np.testing.assert_allclose(calls[0][0], [np.pi/2, np.pi/2])
        self.assertEqual(calls[0][1], 9)

    def test_scene_similarity_preserves_projection(self):
        ds = data.CloudDataset(self.root, normalize_scene=True)
        cloud = ds.read_point_cloud()
        self.assertGreaterEqual(float(cloud.points.min()), .05 - 1e-6)
        self.assertLessEqual(float(cloud.points.max()), .95 + 1e-6)
        frame = ds.train[0]
        raw = data.camera_matrices(frame)
        norm = ds.camera_matrices(frame)
        points = np.array([[0.,0.,-3.],[1.,1.,-2.],[-1.,2.,-5.]])
        raw_clip = np.column_stack([points,np.ones(len(points))]) @ raw['full_proj_transform']
        norm_clip = np.column_stack([ds.normalize_points(points),np.ones(len(points))]) @ norm['full_proj_transform']
        np.testing.assert_allclose(raw_clip[:,:3]/raw_clip[:,3:], norm_clip[:,:3]/norm_clip[:,3:], atol=1e-6)
        np.testing.assert_array_equal(frame.sun_direction, self.ds.train[0].sun_direction)
        np.testing.assert_allclose(ds.normalized_camera_to_world(frame)[:3,:3], frame.camera_to_world[:3,:3])
        self.assertEqual(ds.manifest()['scene_similarity']['original_bbox'], [[-2.,-1.,-3.],[2.,1.,3.]])

    def test_resize_and_overlap_rejected(self):
        smaller = data.CloudDataset(self.root, resolution=4, cache_images=0)
        self.assertEqual(smaller.image_uint8(smaller.train[0]).shape, (2,4,3))
        self.assertEqual(smaller.cache_bytes, 0)
        meta = json.loads((self.root/'transforms_test.json').read_text())
        meta['frames'] = [json.loads((self.root/'transforms_train.json').read_text())['frames'][0]]
        (self.root/'transforms_test.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            data.CloudDataset(self.root)


def audit_dataset(path):
    ds = data.CloudDataset(path, resolution=64, cache_images=2)
    manifest = ds.manifest()
    assert (len(ds.train), len(ds.test), len(ds.stage1)) == (1308, 152, 73)
    assert (manifest['sun_count'], manifest['train_sun_count']) == (60, 56)
    assert (manifest['test_unseen_sun_count'], manifest['test_seen_sun_new_combination_count']) == (96,56)
    for frame in [ds.train[0], ds.test[0], ds.stage1[0]]:
        rgb = ds.image(frame)
        assert rgb.dtype == np.float32 and np.isfinite(rgb).all() and 0 <= rgb.min() <= rgb.max() <= 1
    assert len(ds._images) <= 2
    summary = {k: v for k, v in manifest.items() if k not in ('train_files', 'test_files', 'stage1_members')}
    summary['cache_bytes_after_sample_reads'] = ds.cache_bytes
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append', default=[])
    args, rest = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0]]+rest, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    for path in args.dataset:
        audit_dataset(path)
