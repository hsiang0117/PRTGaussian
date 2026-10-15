# Headless PRTGaussian import/checkpoint patch

Target upstream: `zhanglbthu/PRTGaussian`, commit `f29bbf1a03416aecf879782906b3be377f84a4f8`.

Apply from the repository root:

```bash
git apply --check cloudgs_adapter/upstream_patches/0001-headless-imports-and-albedo-checkpoint.patch
git apply cloudgs_adapter/upstream_patches/0001-headless-imports-and-albedo-checkpoint.patch
```

The adapter may use a different installation directory; substitute the actual patch path. The patch itself changes three upstream files only:

- `scene/__init__.py`: load legacy dataset/camera/light helpers only when constructing `Scene`. Importing `scene.gaussian_model` still executes the package initializer, but it no longer imports the EXR dataset loader. Existing callers constructing `Scene` continue to require its legacy dependencies.
- `my_utils/sh/pm2sh_v2.py`: retain only `math` and `torch` at module scope. Move the image-export demo's NumPy, imageio, and torchvision dependencies into `__main__`; remove unused PIL, transforms, pyshtools, sys, and OpenEXR imports. Every SH function body remains identical.
- `scene/gaussian_model.py`: capture `_albedo`, the parameter used by initialization, the optimizer, PLY saving, and restore, instead of the unrelated empty `albedo` attribute. This fixes checkpoint restoration; it changes no rendering or optimizer formula.

## Runtime dependencies of the headless import path

The unchanged `GaussianModel` path requires PyTorch, NumPy, plyfile, commentjson, tinycudann, and the `simple_knn` CUDA extension. The unchanged renderer additionally requires `diff_gaussian_rasterization`. The RGB adapter uses Pillow; progress logging may use tqdm. There is no need to install Open3D, OpenEXR/Imath, pyshtools, matplotlib, scipy, or pymeshlab for this headless path. This statement does not cover the original `train.py`, which still imports Open3D and torchvision itself, or the legacy `Scene` constructor.

Static import closure checked: `scene` → `scene.gaussian_model`, `arguments`, `utils.system_utils`, `utils.sh_utils`, `utils.general_utils`, `utils.graphics_utils`; `gaussian_renderer` uses this same model and torch/math/rasterizer. None of these top-level imports reaches the legacy dataset reader after the patch. `get_sh_coeffs` reaches only torch/math.

## Verification performed locally

- `git apply --check` succeeds against the specified unmodified upstream checkout.
- All three patched modules parse as Python.
- AST comparison confirms every function in `pm2sh_v2.py` is unchanged.
- The audit checkout remains unmodified; only this patch and this note were written.

Native CUDA imports and numerical forward/backward checks must also run in the server venv after compiling its extensions; local static checks do not establish server runtime success.
