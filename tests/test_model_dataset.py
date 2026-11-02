from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_inputs_samples import _record

from fireline.data.cache import materialize, open_cached
from fireline.data.convert import save_fire
from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES, StatsAccumulator
from fireline.models.spread import SpreadModel, spread_loss
from fireline.train.dataset import SpreadDataset, index_samples


def _cached_fire(tmp_path: Path, days: int = 9, h: int = 40, w: int = 36) -> Path:
    record = _record(days=days, h=h, w=w)
    npz = tmp_path / "data" / "2019" / "fire_t.npz"
    save_fire(record, npz)
    return materialize(npz, tmp_path / "cache")


def test_cache_roundtrip_is_memory_mapped(tmp_path: Path) -> None:
    fire = _cached_fire(tmp_path)
    record = open_cached(fire)
    assert isinstance(record.dynamic, np.memmap) and record.dates[0] == "2019-08-01"
    assert materialize(tmp_path / "data" / "2019" / "fire_t.npz", tmp_path / "cache") == fire  # idempotent


def test_dataset_crops_and_full_images(tmp_path: Path) -> None:
    fire = _cached_fire(tmp_path)
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire))
    stats = accumulator.result((2019,))
    refs = index_samples([fire], history=5)
    assert [r.t for r in refs] == [4, 5, 6, 7]
    train = SpreadDataset(refs, stats, history=5, crop=32)
    item = train[0]
    assert item["dynamic"].shape == (5, len(DYNAMIC_FEATURES), 32, 32)
    assert item["static"].shape == (len(STATIC_FEATURES), 32, 32) and item["valid"].all()
    full = SpreadDataset(refs, stats, history=5, crop=None)[0]
    assert full["dynamic"].shape[-2:] == (64, 64) and int(full["valid"].sum()) == 40 * 36
    assert torch.isfinite(full["dynamic"]).all()


def test_model_shapes_and_gradients() -> None:
    model = SpreadModel(history=5)
    dynamic = torch.randn(2, 5, len(DYNAMIC_FEATURES), 64, 32)
    static = torch.randn(2, len(STATIC_FEATURES), 64, 32)
    logits = model(dynamic, static)
    assert logits.shape == (2, 64, 32)
    target = (torch.rand(2, 64, 32) > 0.95).float()
    valid = torch.ones(2, 64, 32, dtype=torch.bool)
    valid[:, :, -4:] = False
    loss = spread_loss(logits, target, valid)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.parameters())


def test_window_reads_match_full_reads(tmp_path: Path) -> None:
    from fireline.data.inputs import raw_dynamic, raw_static

    record = open_cached(_cached_fire(tmp_path))
    window = (slice(5, 21), slice(3, 30))
    np.testing.assert_array_equal(
        raw_dynamic(record, [2, 3, 4], window), raw_dynamic(record, [2, 3, 4])[..., 5:21, 3:30]
    )
    np.testing.assert_array_equal(raw_static(record, window), raw_static(record)[..., 5:21, 3:30])
    np.testing.assert_array_equal(raw_dynamic(record, [1, 3]), raw_dynamic(record, slice(1, 4))[[0, 2]])


def test_cache_disk_check(tmp_path: Path, monkeypatch) -> None:
    import shutil
    from collections import namedtuple

    import pytest

    from fireline.train.supervised import check_cache_disk

    npz = tmp_path / "f.npz"
    npz.write_bytes(b"x" * 1000)
    check_cache_disk([npz], tmp_path / "cache")  # plenty of space
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(10, 10, 100))
    with pytest.raises(OSError, match="Not enough disk"):
        check_cache_disk([npz], tmp_path / "cache")


def test_fire_batches_match_per_sample_dataset(tmp_path: Path) -> None:
    from fireline.eval.fire_batches import iterate_batches

    fire = _cached_fire(tmp_path, days=11, h=40, w=36)
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire))
    stats = accumulator.result((2019,))
    per_sample = SpreadDataset(index_samples([fire], history=5), stats, history=5, crop=None)
    batches = list(iterate_batches([fire], stats, 5, torch.device("cpu"), batch_size=4, workers=0))
    assert [b["dynamic"].shape[0] for b in batches] == [4, 2]
    stacked = {k: torch.cat([b[k] for b in batches]) for k in ("dynamic", "static", "target", "valid")}
    for i in range(len(per_sample)):
        item = per_sample[i]
        for key in ("dynamic", "static", "target", "valid"):
            torch.testing.assert_close(stacked[key][i].to(item[key].dtype), item[key], rtol=1e-6, atol=1e-6)


def test_predict_fire_encodes_each_day_once_and_matches_forward(tmp_path: Path) -> None:
    from fireline.eval.fire_batches import build_fire, predict_fire

    fire_dir = _cached_fire(tmp_path, days=12, h=40, w=36)
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire_dir))
    stats = accumulator.result((2019,))
    model = SpreadModel(history=5).eval()
    fire = build_fire(fire_dir, stats, history=5)
    calls = []
    model.encoder.register_forward_hook(lambda _m, inputs, _o: calls.append(inputs[0].shape[0]))
    outputs = list(predict_fire(model, fire, torch.device("cpu"), batch_size=3, day_chunk=4))
    assert sum(calls) == 11  # input days 0..10 encoded exactly once (targets t = 4..10; day 11 is only a target)
    per_sample = SpreadDataset(index_samples([fire_dir], history=5), stats, history=5, crop=None)
    flat = [(t, p) for ts, probs in outputs for t, p in zip(ts, probs, strict=True)]
    assert [t for t, _ in flat] == [r.t for r in per_sample.refs]
    with torch.no_grad():
        for i, (_, prob) in enumerate(flat):
            item = per_sample[i]
            expected = torch.sigmoid(model(item["dynamic"][None], item["static"][None]))[0]
            torch.testing.assert_close(prob, expected, rtol=1e-4, atol=1e-5)
