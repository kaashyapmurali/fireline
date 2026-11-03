"""Model-derived architecture facts and non-invasive feature capture for the report."""

from __future__ import annotations

from contextlib import ExitStack

import torch
from torch import nn

from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES
from fireline.models.spread import FIRE_CHANNEL, SpreadModel


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def architecture_facts(model: SpreadModel, crop_px: int = 128) -> dict:
    """Shapes for B=1 (batch omitted), from actual convolution dimensions/strides and module parameters."""
    size, scales = crop_px, []
    for i, (stage, fuse) in enumerate(zip(model.encoder.stages, model.fuse, strict=True)):
        conv = stage.body[0]
        stride = conv.stride[0]
        size = (size + 2 * conv.padding[0] - conv.kernel_size[0]) // stride + 1
        scales.append({
            "index": i, "name": ("full", "half", "quarter", "coarse")[i], "factor": 2 ** i,
            "pixel_m": 375 * 2 ** i, "cin": conv.in_channels, "channels": conv.out_channels, "stride": stride,
            "encoder_shape": [model.history, conv.out_channels, size, size],
            "shape": [conv.out_channels, size, size], "parameters": count(stage),
            "fusion_in": fuse[0].in_channels, "fusion_out": fuse[0].out_channels, "fusion_parameters": count(fuse),
        })
    decoder = []
    for i in range(len(model.up) - 1, -1, -1):
        block = model.up[i]
        decoder.append({"index": i, "name": scales[i]["name"], "up_channels": scales[i + 1]["channels"],
                        "skip_channels": scales[i]["channels"], "concat_channels": block.body[0].in_channels,
                        "shape": scales[i]["shape"], "parameters": count(block)})
    first = model.encoder.stages[0]
    norm_parameters = sum(count(m) for m in first.modules() if isinstance(m, nn.GroupNorm))
    return {
        "crop_px": crop_px, "history": model.history, "batch_omitted": True,
        "dynamic_shape": [model.history, len(DYNAMIC_FEATURES) + model.extra_channels, crop_px, crop_px],
        "static_shape": [len(STATIC_FEATURES), crop_px, crop_px],
        "folded_shape": [model.history, first.body[0].in_channels, crop_px, crop_px],
        "scales": scales, "decoder": decoder,
        "head": {"mask_shape": [1, crop_px, crop_px], "concat_channels": model.head[0].body[0].in_channels,
                 "features_shape": [model.head[1].in_channels, crop_px, crop_px],
                 "logit_shape": [model.head[1].out_channels, crop_px, crop_px], "probability_shape": [crop_px, crop_px],
                 "block_parameters": count(model.head[0]), "projection_parameters": count(model.head[1])},
        "first_block_count": {"conv1": count(first.body[0]), "conv2": count(first.body[3]),
                              "projection": count(first.skip[0]) if not isinstance(first.skip, nn.Identity) else 0,
                              "normalisation": norm_parameters},
        "budget": {"encoder": count(model.encoder), "fusion": count(model.fuse), "decoder": count(model.up),
                   "head": count(model.head), "total": count(model)},
    }


def capture_decode(model: SpreadModel, encoded: list[torch.Tensor], last_fire: torch.Tensor
                   ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run the real decode method unchanged, observing each fusion/whole decoder-block output with removable hooks."""
    fused, decoded = {}, {}

    def capture(target: dict, index: int):
        def hook(_module, _inputs, output):
            target[index] = output.detach()
        return hook

    with ExitStack() as stack:
        for i, module in enumerate(model.fuse):
            stack.enter_context(module.register_forward_hook(capture(fused, i)))
        for i, module in enumerate(model.up):
            stack.enter_context(module.register_forward_hook(capture(decoded, i)))
        logits = model.decode(encoded, last_fire)
    return logits, fused, decoded


def last_fire_channel(dynamic: torch.Tensor) -> torch.Tensor:
    return dynamic[:, -1, FIRE_CHANNEL:FIRE_CHANNEL + 1]
