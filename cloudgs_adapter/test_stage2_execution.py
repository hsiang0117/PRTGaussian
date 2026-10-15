"""CPU checks for batch reuse, precision boundaries, and optimizer equivalence."""
import copy
from types import SimpleNamespace
import unittest

import torch

try:
    from .stage2_execution import stage2_backward
except ImportError:
    from stage2_execution import stage2_backward


class Decoder(torch.nn.Module):
    def __init__(self, half_output=False):
        super().__init__()
        self.linear = torch.nn.Linear(3, 243)
        self.half_output = half_output
        self.calls = 0

    def forward(self, xyz):
        self.calls += 1
        value = self.linear(xyz)
        return value.half() if self.half_output else value


class Stage2ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(728)
        self.xyz = torch.nn.Parameter(torch.rand(7, 3))
        self.opacity = torch.nn.Parameter(torch.rand(7, 1))
        self.decoder = Decoder()
        self.frames = [SimpleNamespace(sun_index=i) for i in range(8)]
        self.coefficients = {i: torch.randn(3, 81) for i in range(8)}
        self.targets = [torch.rand(7, 3) for _ in range(8)]

    def run_variant(self, mode, half_output=False):
        xyz = torch.nn.Parameter(self.xyz.detach().clone())
        opacity = torch.nn.Parameter(self.opacity.detach().clone())
        decoder = copy.deepcopy(self.decoder)
        decoder.half_output = half_output
        parameters = [xyz, opacity, *decoder.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        def render_color(frame, color):
            return color * opacity + 0.03 * xyz, self.targets[frame.sun_index]
        loss = stage2_backward(self.frames, SimpleNamespace(get_xyz=xyz), decoder,
                               self.coefficients, render_color, mode=mode)
        gradients = [p.grad.detach().clone() for p in parameters]
        optimizer.step()
        return loss, gradients, [p.detach().clone() for p in parameters], decoder.calls

    def test_shared_per_frame_matches_loss_gradients_and_adam(self):
        for half in (False, True):
            with self.subTest(half_output=half):
                reference = self.run_variant("reference", half)
                shared = self.run_variant("shared_per_frame", half)
                self.assertAlmostEqual(reference[0], shared[0], places=6)
                for expected, actual in zip(reference[1], shared[1]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for expected, actual in zip(reference[2], shared[2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(reference[3], 8)
                self.assertEqual(shared[3], 1)

    def test_single_backward_changes_half_precision_accumulation(self):
        reference = self.run_variant("reference", half_output=True)
        shared = self.run_variant("shared_sum", half_output=True)
        self.assertAlmostEqual(reference[0], shared[0], places=6)
        for expected, actual in zip(reference[1][:2], shared[1][:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # This is why the production path retains per-image network backward.
        self.assertGreater(max(float((a-b).abs().max())
                               for a, b in zip(reference[1][2:], shared[1][2:])), 0)

    def test_transport_does_not_add_a_position_gradient(self):
        decoder = copy.deepcopy(self.decoder)
        def render_color(frame, color):
            return color * self.opacity, self.targets[frame.sun_index]
        stage2_backward(self.frames, SimpleNamespace(get_xyz=self.xyz), decoder,
                        self.coefficients, render_color)
        self.assertIsNone(self.xyz.grad)

    def test_nonfinite_batch_cannot_reach_optimizer_step(self):
        def render_color(frame, color):
            return color, torch.full_like(color, float("nan"))
        with self.assertRaises(FloatingPointError):
            stage2_backward(self.frames, SimpleNamespace(get_xyz=self.xyz),
                            self.decoder, self.coefficients, render_color)


if __name__ == "__main__":
    unittest.main()
