"""CPU-only checks for the headless runner's native SH and batch equivalence.

This deliberately imports no renderer, CUDA extension, or Scene loader.
Run: python -m unittest discover -s cloudgs_adapter -p test_train_logic.py -v
"""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


def native_function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "GaussianModel": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


def upstream_root():
    here = Path(__file__).resolve().parent
    # The server installs this package directly under the PRTGaussian checkout.
    if (here.parent / "train.py").is_file():
        return here.parent
    # Local review layout keeps a separate, unmodified upstream audit checkout.
    return here.parent / "prtgaussian_audit/source"


class TrainLogicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_native_sh_color_and_gradient_equivalence(self):
        native = native_function(upstream_root() / "train.py", "compute_diffuse_colors")
        xyz = torch.rand(11, 3, requires_grad=True)
        gaussian = type("Gaussian", (), {"get_xyz": xyz})()
        network = torch.nn.Linear(3, 243)
        other = copy.deepcopy(network)
        light = torch.rand(3, 81)
        with patch("torch.cuda.is_available", return_value=False):
            expected = native(light, gaussian, network, 9, 9, "not_origin", "NeRF")
        actual = (other(xyz.detach()).float().view(-1, 3, 81) * light).sum(dim=2).contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for a, b in zip(network.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertIsNone(xyz.grad, "Native transport intentionally detaches Gaussian XYZ")

    def test_per_sample_backward_matches_native_batch_mean(self):
        network = torch.nn.Linear(3, 243)
        accumulated = copy.deepcopy(network)
        xyz = torch.rand(11, 3)
        lights = torch.rand(8, 3, 81)
        targets = torch.rand(8, 11, 3)
        losses = []
        for light, target in zip(lights, targets):
            rgb = (network(xyz).view(-1, 3, 81) * light).sum(2)
            losses.append(0.8 * (rgb - target).abs().mean())
        (sum(losses) / len(losses)).backward()
        for light, target in zip(lights, targets):
            rgb = (accumulated(xyz).view(-1, 3, 81) * light).sum(2)
            (0.8 * (rgb - target).abs().mean() / len(lights)).backward()
        for expected, actual in zip(network.parameters(), accumulated.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)

    def test_checkpoint_capture_fix_uses_live_albedo(self):
        path = upstream_root() / "scene/gaussian_model.py"
        source = path.read_text(encoding="utf-8")
        # The local audit stays pristine; server source already contains this patch.
        source = source.replace("            self.albedo,\n", "            self._albedo,\n")
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
        capture = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[capture], type_ignores=[])), str(path), "exec"), namespace)
        class Optimizer:
            def state_dict(self):
                return {}
        gaussian = type("Gaussian", (), {})()
        for field in ["active_sh_degree", "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "max_radii2D", "xyz_gradient_accum", "denom", "spatial_lr_scale"]:
            setattr(gaussian, field, object())
        gaussian.albedo = torch.empty(0)
        gaussian._albedo = torch.rand(4, 3)
        gaussian.optimizer = Optimizer()
        captured = namespace["capture"](gaussian)
        self.assertIs(captured[10], gaussian._albedo)
        self.assertEqual(tuple(captured[10].shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
