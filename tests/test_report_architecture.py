"""Architecture shapes/counts come from the implemented model; capture hooks do not alter its forward pass."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from fireline.models.spread import SpreadModel, encode_days

spec = importlib.util.spec_from_file_location("report_architecture", Path(__file__).parents[1] / "report/scripts/architecture.py")
assert spec is not None and spec.loader is not None
architecture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(architecture)


def test_baseline_architecture_budget_and_shapes() -> None:
    model = SpreadModel(history=5)
    facts = architecture.architecture_facts(model)
    assert facts["dynamic_shape"] == [5, 18, 128, 128]
    assert facts["static_shape"] == [21, 128, 128]
    assert facts["folded_shape"] == [5, 39, 128, 128]
    assert [s["parameters"] for s in facts["scales"]] == [21888, 57728, 230144, 312064]
    assert [s["encoder_shape"] for s in facts["scales"]] == [
        [5, 32, 128, 128], [5, 64, 64, 64], [5, 128, 32, 32], [5, 128, 16, 16]]
    assert [s["fusion_in"] for s in facts["scales"]] == [160, 320, 640, 640]
    assert [s["concat_channels"] for s in facts["decoder"]] == [256, 192, 96]
    assert facts["head"]["concat_channels"] == 33
    assert facts["budget"] == {"encoder": 621824, "fusion": 190496, "decoder": 676160, "head": 20001, "total": 1508481}
    assert sum(v for k, v in facts["budget"].items() if k != "total") == facts["budget"]["total"]
    assert sum(facts["first_block_count"].values()) == facts["scales"][0]["parameters"]


def test_capture_observes_whole_blocks_and_does_not_change_decode() -> None:
    model = SpreadModel(history=5, widths=(8, 16, 24, 32)).eval()
    dynamic, static = torch.randn(1, 5, 18, 32, 32), torch.randn(1, 21, 32, 32)
    with torch.no_grad():
        encoded = encode_days(model.encoder, dynamic, static)
        last_fire = architecture.last_fire_channel(dynamic)
        expected = model.decode(encoded, last_fire)
        actual, fused, decoded = architecture.capture_decode(model, encoded, last_fire)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [tuple(fused[i].shape) for i in range(4)] == [(1, 8, 32, 32), (1, 16, 16, 16), (1, 24, 8, 8), (1, 32, 4, 4)]
    assert [tuple(decoded[i].shape) for i in (2, 1, 0)] == [(1, 24, 8, 8), (1, 16, 16, 16), (1, 8, 32, 32)]
    assert all(not m._forward_hooks for m in [*model.fuse, *model.up])
    facts = architecture.architecture_facts(model, crop_px=32)
    assert facts["scales"][-1]["shape"] == [32, 4, 4]
    assert facts["budget"]["total"] == sum(p.numel() for p in model.parameters())
