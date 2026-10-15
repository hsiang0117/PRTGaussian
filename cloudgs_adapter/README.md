# CloudDataset adapter for PRTGaussian

Upstream: <https://github.com/zhanglbthu/PRTGaussian>, commit
`f29bbf1a03416aecf879782906b3be377f84a4f8`.
This is PRTGaussian, not PRTGS. Its learned SH transport has no viewing-direction
input and is not an explicit participating-medium inverse renderer.

## Protocol

- Read `transforms_train.json`, `transforms_test.json`, PNG and the existing
  `points3d.ply` without overwriting source data. Preserve sparse camera/light
  combinations. Never use final test images for early stopping or initialization.
- Stage 1: for each training camera, decode its available **training** PNGs from
  sRGB, average in linear space, encode to sRGB and quantize to a derived PNG in
  the output directory. This finite average is only an approximation of uniform
  illumination. Standard 3DGS L1+DSSIM, color SH training and densification.
- Stage 2: transfer Stage-1 XYZ only, as native `load_pts` does; reset covariance,
  opacity and color. Preserve native `config/config.json`, 243 RGB SH transport
  coefficients, order 9, batch 8, Adam transport LR 1e-3, joint Gaussian updates,
  released `0.8*L1` loss, no densification and no Stage-2 XYZ LR schedule.
- Batch gradients accumulate sequentially before one optimizer update; this is
  mathematically equivalent to the released batch mean and uses less memory.
- All positions and camera centers undergo the same recorded translation and
  uniform scale, putting the source point bbox in [0.05,0.95] for HashGrid input.
  Rotations and sun directions do not change; camera clipping distances scale too.
  CPU checks verify projection invariance. There is no GT VDB density supervision.
- PNG values are the RGB targets, as for the other CloudDataset comparisons.
  Do not add per-frame exposure, max normalization or extra output gamma. Export
  unclipped float32 NPZ and clipped PNG. No mask loss.
- On the last iteration apply the optimizer update (released loop skips it).
  Fixed requested budgets replace native test-driven early stopping, and final
  checkpoint saving does not depend on an early-stop-only variable.

## Setup on the Ubuntu / RTX 4090 server

Use a separate venv; never reuse another project's installed extension binary.
The original requirements contain conflicting Pillow/OpenEXR pins and are kept
for provenance. `requirements-server.txt` is the headless dependency selection.

```bash
python3 -m venv .venv
# Reuse direct local torch/CUDA wheel paths to avoid pip redownloading CUDA libs.
.venv/bin/python -m pip install /myfiles/projects/Vol3DGS/dependency-cache/wheels/*.whl
.venv/bin/python -m pip install -r cloudgs_adapter/requirements-server.txt
git apply --ignore-space-change cloudgs_adapter/upstream_patches/0001-headless-imports-and-albedo-checkpoint.patch
git apply --ignore-space-change cloudgs_adapter/upstream_patches/0002-simple-knn-cfloat.patch
git apply --ignore-space-change cloudgs_adapter/upstream_patches/0003-rasterizer-cstdint.patch
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export TCNN_CUDA_ARCHITECTURES=89
export MAX_JOBS=4
pip wheel --no-deps --no-build-isolation ./submodules/simple-knn -w dependency-cache/wheels
pip wheel --no-deps --no-build-isolation ./submodules/diff-gaussian-rasterization -w dependency-cache/wheels
# Clone official tiny-cuda-nn and submodules under external-dependencies.
# The server reused clean local Git objects and submodule caches, not binaries.
# Tested tiny-cuda-nn 2.0 source: 3daa6e5b8b8e0fa17971a1a20acfbb3754494b38.
pip wheel --no-deps --no-build-isolation ./external-dependencies/tiny-cuda-nn/bindings/torch -w dependency-cache/wheels
pip install --no-deps dependency-cache/wheels/*.whl
python -m unittest discover -s cloudgs_adapter -p 'test*.py' -v
python -m cloudgs_adapter.probe_cuda
```

Legacy EXR/OpenIllumination loaders are imported lazily. SH formulas and CUDA
rendering algorithms are unmodified. A native checkpoint field typo is fixed:
`capture()` saves the trained `_albedo`, matching `restore()`.
The KNN source also explicitly includes `<cfloat>` for `FLT_MAX`; CUDA 12.8 no
longer happens to supply that macro through its transitive includes. This is a
standard missing-header fix, with no numeric or kernel algorithm changes.
The rasterizer header explicitly includes `<cstdint>` for `std::uintptr_t` and
`uint32_t`/`uint64_t`, for the same CUDA 12.8 header compatibility reason.
The fork already contains these source changes; patch commands are for upstream
checkouts and must not be applied a second time.
`requirements-server.lock.txt` records the actually installed versions. CUDA
extension wheels are rebuilt from the pinned source, not fetched by their generic
`0.0.0` package version from PyPI.

## Short validation (not a formal experiment)

```bash
python -m cloudgs_adapter.train_cloud \
  --source /myfiles/data/CloudDatasetUniform \
  --output output/YYYYMMDD_HHMMSS_cloud_short \
  --resolution 512 --stage1-iterations 300 --stage2-iterations 200 \
  --preview-every 100 --diagnostic
```

Default keeps all source points and the full native transport network. Diagnostic
step counts/resolution must not be presented as the paper's final baseline.
Record finite gradients, fixed training-probe before/after L1, saved model restore
agreement, fixed-camera multi-sun response, CUDA synchronized loop time excluding
preview/export, and peak allocated memory. The output includes cfg_args, PLY,
checkpoint, rendertest PNG/NPZ and a few final test previews. No formal full-test
evaluation is performed by this validation entry.

## Formal 01/02 env-off queue

```bash
python -u -m cloudgs_adapter.run_formal_queue \
  --control /myfiles/projects/PRTGaussian/.experiment-control/20261015_formal_envoff --run
python -m cloudgs_adapter.run_formal_queue \
  --control /myfiles/projects/PRTGaussian/.experiment-control/20261015_formal_envoff --poll
```

The caller detaches the queue using a new process session. The queue runs 01 then
02, each with 30k standard-3DGS mean-image iterations and 30k native SH-transport
iterations at 1024, batch 8, the original full network, and previews every 1000
steps in both stages. Thirty thousand is the released training parser's effective
default; the unused `Scene.iterations=300000` INI key is not an active budget.
Do not describe these counts as an independently verified paper-specified budget.

Each case gets its actual launch timestamp and dataset identifier in `output`.
The queue holds a lock inherited by the training child, refuses an occupied GPU
or less than 10 GiB free disk before a case, records child PIDs and logs, validates
successful budgets/source hashes, and stops on failure without blind retries.

Stage 2 writes a rotating resumable `checkpoint_latest.pt` every 7000 iterations,
using atomic replacement and a free-space guard. It saves both optimizers, the
remaining shuffled training stack and Python/NumPy/Torch random states. Saving
does not change losses, sampling or iteration budgets; checkpoint/preview I/O is
excluded from training-loop timing. After the final checkpoint is safely saved,
the rotating checkpoint is removed. Stage-1 XYZ/PLY and timing are saved when
that stage completes. To resume Stage 2 into a new output directory:

```bash
python -m cloudgs_adapter.train_cloud --source /myfiles/data/CloudDatasetUniform \
  --output output/NEW_TIMESTAMP_prt_cloud01_envoff_resume \
  --stage1-checkpoint output/ORIGINAL_RUN/coarse_xyz.pt \
  --resume-stage2 output/ORIGINAL_RUN/checkpoint_latest.pt \
  --stage1-iterations 30000 --stage2-iterations 30000 \
  --resolution 1024 --batch-size 8 --preview-every 1000 --checkpoint-every 7000
```

Review the queue state before resuming; an existing unfinished queue is not
automatically overwritten. Formal evaluation and dataset/model downloads are
separate from this training launch.
