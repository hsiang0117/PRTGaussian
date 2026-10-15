"""Headless, fixed-split CloudDataset adapter for the official PRTGaussian model.

Run from the repository root: python -m cloudgs_adapter.train_cloud --help.
The SH transport, rasterizer and Stage-2 loss follow the released implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from .cloud_data import CloudDataset


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(value, path):
    """One rotating checkpoint, with space for atomic replacement and 1 GiB spare."""
    path = Path(path)
    seen = set()
    def tensor_bytes(item):
        if isinstance(item, torch.Tensor):
            identity = (str(item.device), item.untyped_storage().data_ptr())
            if identity in seen:
                return 0
            seen.add(identity)
            return item.untyped_storage().nbytes()
        if isinstance(item, dict):
            return sum(tensor_bytes(x) for x in item.values())
        if isinstance(item, (list, tuple)):
            return sum(tensor_bytes(x) for x in item)
        return 0
    required = tensor_bytes(value) + 1024**3
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(f"Insufficient checkpoint disk space: need {required} free bytes")
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in value["torch_cuda"]])


def restore_adam_step_device(state):
    # map_location='cuda' must not move uncapturable Adam's scalar step counters
    # from their original CPU placement; keep its original numerical code path.
    for group in state["param_groups"]:
        if not group.get("capturable", False) and not group.get("fused", False):
            for identifier in group["params"]:
                values = state["state"].get(identifier, {})
                if isinstance(values.get("step"), torch.Tensor):
                    values["step"] = values["step"].cpu()
    return state


def save_image(path, image):
    array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Nonfinite render: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), rgb=array.astype(np.float32))
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def options(iterations):
    # Released OptimizationParams defaults, not shortened LR schedules.
    return SimpleNamespace(
        position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=30000,
        feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.005,
        rotation_lr=0.001, percent_dense=0.01, iterations=iterations,
    )


def sample_snapshot(model):
    fields = ["_xyz", "_opacity", "_scaling", "_rotation"]
    return {k: getattr(model, k).detach()[:1024].clone() for k in fields}


def changes(model, before):
    return {k: float((getattr(model, k).detach()[:len(v)] - v).abs().mean())
            for k, v in before.items() if len(getattr(model, k)) >= len(v)}


def check_gradients(parameters):
    active = 0
    absolute_sum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Nonfinite gradient")
            active += 1
            absolute_sum += float(parameter.grad.detach().abs().mean())
    if not active or not absolute_sum > 0:
        raise FloatingPointError("No nonzero gradients")
    return {"active_tensors": active, "sum_mean_abs": absolute_sum}


def train(args):
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import ssim
    from my_utils.sh.pm2sh_v2 import get_sh_coeffs
    import tinycudann as tcnn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output or ("output/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_cloud_short"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "cfg_args.json", vars(args))
    (output / "cfg_args").write_text(str(args), encoding="utf-8")
    dataset = CloudDataset(args.source, resolution=args.resolution, cache_images=args.cache_images,
                           normalize_scene=True)
    source_paths = [dataset.point_cloud_path] + [Path(args.source) / f"transforms_{split}.json" for split in ("train", "test")]
    source_hashes = {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in source_paths}
    manifest = dataset.manifest()
    write_json(output / "data_manifest.json", manifest)
    dataset.prepare_stage1_means(output / "stage1_training_means")
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device="cuda")
    pcd = dataset.read_point_cloud()
    pcd = BasicPointCloud(pcd.points, pcd.colors, pcd.normals)
    if args.point_limit:
        # Only diagnostic opt-in; the default keeps all source points.
        indices = np.random.default_rng(args.seed).choice(len(pcd.points), min(args.point_limit, len(pcd.points)), replace=False)
        pcd = BasicPointCloud(*(np.asarray(x)[indices] for x in pcd))
    camera_centers = np.array([dataset.camera(f, device="cpu", include_image=False).camera_center.numpy()
                               for f in dataset.stage1])
    extent = float(np.max(np.linalg.norm(camera_centers - camera_centers.mean(0), axis=1)) * 1.1)
    model = GaussianModel(3)
    model.create_from_pcd(pcd, extent)
    summary = {"diagnostic_only": args.diagnostic, "source_points": len(pcd.points),
               "stages": {}, "cuda": torch.version.cuda, "torch": torch.__version__,
               "device": torch.cuda.get_device_name(0), "resolution": args.resolution}
    log_path = output / "traininglog.jsonl"

    def log(value):
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)
        write_json(output / "progress.json", {**value, "updated": datetime.now().isoformat()})

    def render_frame(frame, decoder=None, coefficients=None):
        camera = dataset.camera(frame, device="cuda")
        color = None
        if decoder is not None:
            # Same released color/total SH order: RGB coefficients for all 81 bases.
            transfer = decoder(model.get_xyz.detach()).float().view(-1, 3, 81)
            color = (transfer * coefficients).sum(dim=2).contiguous()
        image = render(camera, model, pipe, background, override_color=color)["render"]
        return image, camera.original_image

    probe_stage1 = dataset.stage1[0]
    with torch.no_grad():
        image, target = render_frame(probe_stage1)
        s1_before = float((image - target).abs().mean())
        save_image(output / "rendertest/stage1_000000.png", image)
        save_image(output / "rendertest/stage1_target.png", target)
    if args.stage1_checkpoint:
        state = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        xyz = state["xyz"]
        for key in ("root", "point_cloud_path", "scene_similarity", "train_files", "test_files"):
            if state["data_manifest"][key] != manifest[key]:
                raise ValueError(f"Coarse checkpoint data mismatch: {key}")
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not torch.isfinite(xyz).all():
            raise ValueError("Invalid coarse point cloud")
        if state["steps"] != args.stage1_iterations:
            raise ValueError("Coarse checkpoint does not match the requested Stage-1 budget")
        summary["stage1_reused"] = str(Path(args.stage1_checkpoint).absolute())
    else:
        opt = options(args.stage1_iterations)
        model.training_setup(opt)
        # Stage 1 is standard 3DGS, as described in the paper. The released
        # model's Stage-2 setup intentionally excludes view/color SH features.
        model.optimizer.add_param_group({"params": [model._features_dc], "lr": opt.feature_lr, "name": "f_dc"})
        model.optimizer.add_param_group({"params": [model._features_rest], "lr": opt.feature_lr / 20, "name": "f_rest"})
        model._albedo.requires_grad_(False)
        model.optimizer.param_groups[:] = [g for g in model.optimizer.param_groups if g["name"] != "albedo"]
        stack = []
        first_grad = None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        excluded = 0.0
        for step in range(1, args.stage1_iterations + 1):
            model.update_learning_rate(step)
            if step % 1000 == 0:
                model.oneupSHdegree()
            if not stack:
                stack = list(dataset.stage1)
                random.shuffle(stack)
            frame = stack.pop()
            camera = dataset.camera(frame, device="cuda")
            package = render(camera, model, pipe, background)
            image = package["render"]
            l1 = (image - camera.original_image).abs().mean()
            loss = 0.8 * l1 + 0.2 * (1 - ssim(image, camera.original_image))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage1 loss at {step}")
            loss.backward()
            if step == 1:
                first_grad = check_gradients([p for group in model.optimizer.param_groups for p in group["params"]])
            with torch.no_grad():
                if step < 15000:
                    visible = package["visibility_filter"]
                    model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], package["radii"][visible])
                    model.add_densification_stats(package["viewspace_points"], visible)
                    if step > 500 and step % 100 == 0:
                        model.densify_and_prune(0.0002, 0.005, extent, 20 if step > 3000 else None)
                    if step % 3000 == 0:
                        model.reset_opacity()
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % args.log_every == 0:
                log({"stage": 1, "step": step, "l1": float(l1.detach()), "loss": float(loss.detach()), "gaussians": len(model.get_xyz)})
            if step % args.preview_every == 0 or step == args.stage1_iterations:
                torch.cuda.synchronize()
                pause = time.perf_counter()
                with torch.no_grad():
                    image, _ = render_frame(probe_stage1)
                    save_image(output / f"rendertest/stage1_{step:06d}.png", image)
                torch.cuda.synchronize()
                excluded += time.perf_counter() - pause
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start - excluded
        with torch.no_grad():
            image, target = render_frame(probe_stage1)
            after = float((image - target).abs().mean())
        xyz = model.get_xyz.detach().cpu()
        torch.save({"xyz": xyz, "data_manifest": manifest, "steps": args.stage1_iterations}, output / "coarse_xyz.pt")
        # Albedo is unused in coarse standard-3DGS; densification changes N.
        model._albedo = torch.nn.Parameter(torch.zeros_like(model.get_xyz))
        model.save_ply(str(output / "stage1/point_cloud.ply"))
        summary["stages"]["stage1"] = {"steps": args.stage1_iterations, "training_seconds": seconds,
            "probe_l1_before": s1_before, "probe_l1_after": after, "first_grad": first_grad,
            "gaussians": len(xyz), "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        write_json(output / "stage1_summary.json", summary["stages"]["stage1"])
        del package, loss, l1, image, target, camera
    # Released Scene(load_pts=...) only transfers XYZ, resets all other properties.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    pcd2 = BasicPointCloud(xyz.numpy(), np.full_like(xyz.numpy(), 0.5), np.zeros_like(xyz.numpy()))
    model = GaussianModel(3)
    model.create_from_pcd(pcd2, extent)
    model.training_setup(options(args.stage2_iterations))
    config = json.loads(Path(args.network_config).read_text())
    decoder = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=243,
                  encoding_config=config["encoding"], network_config=config["network"]).cuda()
    write_json(output / "network_config.json", config)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    train_suns = {frame.sun_index: frame for frame in dataset.train}
    coeffs = {sun: dataset.light_coefficients(frame, get_sh_coeffs, order=9).cuda()
              for sun, frame in train_suns.items()}
    probe = dataset.train[0]
    resumed = None
    if args.resume_stage2:
        del model, decoder, decoder_opt
        gc.collect()
        torch.cuda.empty_cache()
        resumed = torch.load(args.resume_stage2, map_location="cuda", weights_only=False)
        for key in ("root", "point_cloud_path", "scene_similarity", "train_files", "test_files"):
            if resumed["data_manifest"][key] != manifest[key]:
                raise ValueError(f"Resume data mismatch: {key}")
        if (resumed["network_config"] != config or resumed["args"]["batch_size"] != args.batch_size
                or resumed["args"]["resolution"] != args.resolution):
            raise ValueError("Resume network, resolution or batch size differs")
        if resumed["args"]["stage2_iterations"] != args.stage2_iterations:
            raise ValueError("Resume must preserve the original Stage-2 budget")
        if not 0 < resumed["step"] < args.stage2_iterations:
            raise ValueError("Resume checkpoint must precede the final iteration")
        if resumed["source_sha256"] != source_hashes:
            raise ValueError("Resume source files changed")
        restore_adam_step_device(resumed["gaussians"][11])
        restore_adam_step_device(resumed["decoder_optimizer"])
        model = GaussianModel(3)
        model.restore(resumed["gaussians"], options(args.stage2_iterations))
        decoder = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=243,
                      encoding_config=config["encoding"], network_config=config["network"]).cuda()
        decoder.load_state_dict(resumed["decoder"])
        decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        decoder_opt.load_state_dict(resumed["decoder_optimizer"])
        summary["stages"] = resumed["summary"]["stages"]
        summary["resumed_from"] = str(Path(args.resume_stage2).absolute())
    initial = sample_snapshot(model)
    network_initial = next(decoder.parameters()).detach().flatten()[:4096].clone()
    with torch.no_grad():
        image, target = render_frame(probe, decoder, coeffs[probe.sun_index])
        s2_before = float((image - target).abs().mean())
        save_image(output / "rendertest/stage2_000000.png", image)
        save_image(output / "rendertest/stage2_target.png", target)
    stack, first_grad, first_step, elapsed_before = [], None, 0, 0.0
    if resumed is not None:
        frames_by_id = {frame.uid: frame for frame in dataset.train}
        stack = [frames_by_id[uid] for uid in resumed["sampler_stack"]]
        first_grad, first_step = resumed["first_grad"], resumed["step"]
        elapsed_before, s2_before = resumed["training_seconds"], resumed["s2_before"]
        restore_rng(resumed["rng"])
        del resumed
    def save_stage2_checkpoint(step, seconds, destination):
        atomic_checkpoint({"gaussians": model.capture(), "decoder": decoder.state_dict(),
            "decoder_optimizer": decoder_opt.state_dict(), "step": step,
            "network_config": config, "data_manifest": manifest, "args": vars(args),
            "source_sha256": source_hashes, "summary": summary, "s2_before": s2_before,
            "first_grad": first_grad, "training_seconds": seconds,
            "sampler_stack": [frame.uid for frame in stack], "rng": capture_rng()}, destination)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    excluded = 0.0
    for step in range(first_step + 1, args.stage2_iterations + 1):
        loss_value = 0.0
        for _ in range(args.batch_size):
            if not stack:
                stack = list(dataset.train)
                random.shuffle(stack)
            frame = stack.pop()
            image, target = render_frame(frame, decoder, coeffs[frame.sun_index])
            # Match released training: only (1-lambda_dssim)*L1, no DSSIM term.
            loss = 0.8 * (image - target).abs().mean() / args.batch_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage2 loss at {step}")
            loss.backward()
            loss_value += float(loss.detach())
        if step == 1:
            first_grad = {"network": check_gradients(decoder.parameters()),
                          "geometry": check_gradients([model._xyz, model._scaling, model._rotation, model._opacity])}
        model.optimizer.step()
        decoder_opt.step()
        model.optimizer.zero_grad(set_to_none=True)
        decoder_opt.zero_grad(set_to_none=True)
        # No densification or LR scheduling in the released Stage-2 train loop.
        if step == 1 or step % args.log_every == 0:
            log({"stage": 2, "step": step, "loss": loss_value, "gaussians": len(model.get_xyz)})
        if step % args.preview_every == 0 or step == args.stage2_iterations:
            torch.cuda.synchronize()
            pause = time.perf_counter()
            with torch.no_grad():
                image, _ = render_frame(probe, decoder, coeffs[probe.sun_index])
                save_image(output / f"rendertest/stage2_{step:06d}.png", image)
            torch.cuda.synchronize()
            excluded += time.perf_counter() - pause
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.stage2_iterations:
            torch.cuda.synchronize()
            pause = time.perf_counter()
            elapsed = elapsed_before + pause - start - excluded
            save_stage2_checkpoint(step, elapsed, output / "checkpoint_latest.pt")
            torch.cuda.synchronize()
            excluded += time.perf_counter() - pause
    torch.cuda.synchronize()
    seconds = elapsed_before + time.perf_counter() - start - excluded
    with torch.no_grad():
        image, target = render_frame(probe, decoder, coeffs[probe.sun_index])
        after = float((image - target).abs().mean())
    summary["stages"]["stage2"] = {"steps": args.stage2_iterations, "batch_size": args.batch_size,
        "training_seconds": seconds, "probe_l1_before": s2_before, "probe_l1_after": after,
        "gaussians": len(model.get_xyz), "first_grad": first_grad,
        "parameter_changes": changes(model, initial),
        "network_mean_abs_change_first_4096": float((next(decoder.parameters()).detach().flatten()[:4096] - network_initial).abs().mean()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    model.save_ply(str(output / "point_cloud/iteration_final/point_cloud.ply"))
    save_stage2_checkpoint(args.stage2_iterations, seconds, output / "checkpoint.pt")
    (output / "checkpoint_latest.pt").unlink(missing_ok=True)
    # Verify real serialization/restoration, not merely successful file creation.
    with torch.no_grad():
        reference, _ = render_frame(probe, decoder, coeffs[probe.sun_index])
        reference = reference.cpu()
    del model, decoder, decoder_opt, loss, image, target, initial, network_initial
    gc.collect()
    torch.cuda.empty_cache()
    loaded = torch.load(output / "checkpoint.pt", map_location="cuda", weights_only=False)
    model = GaussianModel(3)
    model.restore(loaded["gaussians"], options(args.stage2_iterations))
    decoder = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=243,
                  encoding_config=config["encoding"], network_config=config["network"]).cuda()
    decoder.load_state_dict(loaded["decoder"])
    del loaded
    with torch.no_grad():
        restored, _ = render_frame(probe, decoder, coeffs[probe.sun_index])
        restore_error = float((restored.cpu() - reference).abs().max())
    summary["checkpoint_restore_max_abs_error"] = restore_error
    if restore_error > 1e-5:
        raise AssertionError(f"Checkpoint restore changed rendered RGB: {restore_error}")
    # Fixed view / varied sun, selected only by metadata, after all optimization.
    by_sun = {frame.sun_index: frame for frame in dataset.train}
    suns = sorted(by_sun)[::max(1, len(by_sun) // 5)][:5]
    response = []
    with torch.no_grad():
        for sun in suns:
            image, _ = render_frame(probe, decoder, coeffs[sun])
            save_image(output / f"light_response/cam{probe.camera_index:03d}_sun{sun:03d}.png", image)
            response.append(image.cpu())
    summary["light_response"] = {"camera": probe.camera_index, "sun_ids": suns,
        "mean_abs_difference_to_first": [float((x-response[0]).abs().mean()) for x in response]}
    # A few held-out outputs verify the rendering path only, never early stopping.
    with torch.no_grad():
        for frame in dataset.test[::max(1, len(dataset.test) // 4)][:4]:
            coefficient = dataset.light_coefficients(frame, get_sh_coeffs, order=9).cuda()
            image, target = render_frame(frame, decoder, coefficient)
            save_image(output / f"test_preview/{frame.key}.png", image)
            save_image(output / f"test_preview/{frame.key}_gt.png", target)
    summary["status"] = "complete"
    write_json(output / "progress.json", {"stage": "complete", "status": "complete",
               "steps": [args.stage1_iterations, args.stage2_iterations], "updated": datetime.now().isoformat()})
    final_hashes = {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in source_paths}
    if final_hashes != source_hashes:
        raise AssertionError("Source PLY or fixed split changed")
    summary["source_sha256_before_equals_after"] = True
    summary["source_sha256"] = final_hashes
    write_json(output / "validation_summary.json", summary)
    print("COMPLETE " + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--stage1-iterations", type=int, default=30000)
    parser.add_argument("--stage2-iterations", type=int, default=30000)
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--resume-stage2")
    parser.add_argument("--checkpoint-every", type=int, default=7000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--network-config", default="config/config.json")
    parser.add_argument("--point-limit", type=int, default=0)
    parser.add_argument("--cache-images", type=int, default=16)
    parser.add_argument("--preview-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if min(args.stage1_iterations, args.stage2_iterations, args.batch_size, args.preview_every, args.log_every) <= 0:
        parser.error("Iterations, batch size and intervals must be positive")
    if not args.diagnostic and args.point_limit:
        parser.error("--point-limit requires --diagnostic")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if args.resume_stage2 and not args.stage1_checkpoint:
        parser.error("--resume-stage2 also requires the originating --stage1-checkpoint coarse_xyz.pt")
    train(args)


if __name__ == "__main__":
    main()
