"""Execution-only variants of the released Stage-2 objective.

All variants detach XYZ at the transport decoder and average the same per-image
0.8 * L1 losses. Rasterizer graphs are released after each image. The default
shares the decoder forward, but deliberately keeps one decoder backward per
image: casting each image's output derivative to tiny-cuda-nn's internal FP16
before accumulation is part of its existing numerical execution.
"""
from __future__ import annotations

import math

import torch


MODES = ("reference", "shared_per_frame", "shared_sum")


def stage2_backward(frames, model, decoder, coefficients, render_color,
                    mode="shared_per_frame"):
    """Accumulate one native batch's gradients and return its scalar loss.

    ``render_color(frame, color)`` returns ``(RGB, target)``. This function never
    steps or clears either optimizer. ``reference`` reproduces the original
    forward/backward and scalar checks. ``shared_sum`` is an experimental
    validation candidate; use it only after checking low-precision gradients.
    The optimized paths read one host scalar per batch, before optimizer steps.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown Stage-2 execution mode: {mode}")
    frames = tuple(frames)
    if not frames:
        raise ValueError("Stage-2 batch must contain at least one image")
    batch_size = len(frames)
    total = None
    host_total = 0.0
    transfer = None
    leaf = None
    if mode != "reference":
        transfer = decoder(model.get_xyz.detach()).float().view(-1, 3, 81)
        # Separating the rasterizer from the network lets each large image
        # graph be freed immediately, while retaining only one network graph.
        leaf = transfer.detach().requires_grad_(True)
    for index, frame in enumerate(frames):
        if mode == "reference":
            current = decoder(model.get_xyz.detach()).float().view(-1, 3, 81)
        else:
            current = leaf
        color = (current * coefficients[frame.sun_index]).sum(dim=2).contiguous()
        image, target = render_color(frame, color)
        loss = 0.8 * (image - target).abs().mean() / batch_size
        if mode == "reference" and not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage-2 per-image loss")
        loss.backward()
        if mode == "reference":
            host_total += float(loss.detach())
        else:
            total = loss.detach().clone() if total is None else total.add_(loss.detach())
            if mode == "shared_per_frame":
                if leaf.grad is None:
                    raise RuntimeError("Rasterizer did not produce a color derivative")
                # Preserve the original per-image FP32 -> internal FP16 cast,
                # and its sequence of parameter-gradient accumulations.
                transfer.backward(leaf.grad, retain_graph=index + 1 < batch_size)
                leaf.grad = None
        del loss, image, target, color, current
    if mode == "shared_sum":
        if leaf.grad is None:
            raise RuntimeError("Rasterizer did not produce a color derivative")
        transfer.backward(leaf.grad)
    if mode != "reference":
        host_total = float(total)
    # Each per-image L1 is nonnegative, so a finite sum also verifies all loss
    # values. Nonfinite batches raise before either optimizer can update.
    if not math.isfinite(host_total):
        raise FloatingPointError("Nonfinite Stage-2 batch loss")
    return host_total
