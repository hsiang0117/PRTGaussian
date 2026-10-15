"""Small forward/backward check of this checkout's independently built extensions."""
import json
from types import SimpleNamespace

import numpy as np
import torch
import tinycudann as tcnn

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
from my_utils.sh.pm2sh_v2 import get_sh_coeffs


def main():
    torch.manual_seed(0)
    xyz = np.random.default_rng(0).normal(0, .1, (64, 3)).astype(np.float32)
    xyz[:, 2] += 3
    model = GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(xyz, np.full_like(xyz, .5), np.zeros_like(xyz)), 1)
    view = torch.eye(4, device="cuda")
    projection = getProjectionMatrix(.01, 100., 1., 1.).T.cuda()
    camera = SimpleNamespace(FoVx=1., FoVy=1., image_width=64, image_height=64,
        world_view_transform=view, full_proj_transform=view @ projection,
        camera_center=torch.zeros(3, device="cuda"))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    decoder = tcnn.NetworkWithInputEncoding(3, 243,
        {"otype":"HashGrid", "n_levels":2, "n_features_per_level":2,
         "log2_hashmap_size":10, "base_resolution":16, "per_level_scale":1.5},
        {"otype":"CutlassMLP", "activation":"ReLU", "output_activation":"None",
         "n_neurons":32, "n_hidden_layers":2}).cuda()
    light = get_sh_coeffs([.4, .7], order=9).cuda()
    rgb = (decoder((model.get_xyz.detach()-torch.tensor([0.,0.,3.],device="cuda"))+.5).float().view(-1,3,81)*light).sum(2)
    package = render(camera, model, pipe, torch.zeros(3,device="cuda"), override_color=rgb.contiguous())
    image = package["render"]
    (image-.1).square().mean().backward()
    torch.cuda.synchronize()
    for parameter in (model._xyz, model._opacity, next(decoder.parameters())):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(image).all() and package["visibility_filter"].sum() > 0
    print(json.dumps({"status":"pass", "torch":torch.__version__, "cuda":torch.version.cuda,
        "device":torch.cuda.get_device_name(), "visible":int(package["visibility_filter"].sum()),
        "image_min":float(image.min().detach()), "image_max":float(image.max().detach())},indent=2))


if __name__ == "__main__":
    main()
