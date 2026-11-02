from __future__ import annotations

from pathlib import Path

import torch
from test_model_dataset import _cached_fire

from fireline.data.cache import open_cached
from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES, StatsAccumulator
from fireline.eval.fire_batches import build_fire
from fireline.models.jepa import (
    SpreadJEPA,
    effective_rank,
    momentum_schedule,
    near_fire_cells,
    variance_covariance,
)
from fireline.models.spread import FIRE_CHANNEL
from fireline.train.dataset import SpreadDataset, index_samples
from fireline.train.jepa import gate_statistics


def _batch(b: int = 2, h: int = 64, w: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    dynamic = torch.randn(b, 6, len(DYNAMIC_FEATURES), h, w)
    dynamic[:, :, FIRE_CHANNEL] = 0
    dynamic[:, -2:, FIRE_CHANNEL, 20:24, 10:14] = 1
    return dynamic, torch.randn(b, len(STATIC_FEATURES), h, w)


def test_loss_trains_only_online_parts_and_ema_moves_target() -> None:
    model = SpreadJEPA(history=5)
    dynamic, static = _batch()
    loss, stats = model.loss(dynamic, static)
    loss.backward()
    assert torch.isfinite(loss) and stats["near_fire_cells"] > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())
    before = next(model.target_encoder.parameters()).clone()
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1.0)
    model.update_target(0.9)
    after = next(model.target_encoder.parameters())
    torch.testing.assert_close(after, before + 0.1)


def test_near_fire_cells_and_schedule() -> None:
    today = torch.zeros(1, 64, 64, dtype=torch.bool)
    today[0, 0, 0] = True
    cells = near_fire_cells(today, torch.zeros_like(today), radius=1)
    assert cells.shape == (1, 8, 8) and cells.sum() == 4  # cell (0,0) and its 3 neighbours inside the image
    assert momentum_schedule(0, 100) == 0.996 and momentum_schedule(100, 100) == 1.0


def test_collapse_measures() -> None:
    collapsed = torch.ones(500, 16) + 1e-3 * torch.randn(500, 16)
    healthy = torch.randn(500, 16)
    assert effective_rank(healthy) > 12 and variance_covariance(collapsed)[0] > 0.9
    assert variance_covariance(healthy)[0] < 0.1


def test_jepa_dataset_and_gate_statistics(tmp_path: Path) -> None:
    fire_dir = _cached_fire(tmp_path, days=10, h=40, w=36)
    accumulator = StatsAccumulator()
    accumulator.add(open_cached(fire_dir))
    stats = accumulator.result((2019,))
    item = SpreadDataset(index_samples([fire_dir], 5), stats, 5, crop=32, next_day=True)[0]
    assert item["dynamic"].shape[0] == 6
    model = SpreadJEPA(history=5)
    gates = gate_statistics(model, [build_fire(fire_dir, stats, 5)], torch.device("cpu"), amp=False)
    assert 1 <= gates["effective_rank"] <= gates["latent_dim"] == 128
    assert gates["cells"] > 0 and gates["prediction_over_copy_near_fire"] > 0


def test_predictor_starts_as_copy_of_today() -> None:
    from fireline.models.jepa import Predictor, channel_layer_norm

    predictor = Predictor(history=5, latent=16, width=32, depth=1)
    context, forecast = torch.randn(2, 5, 16, 8, 8), torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(predictor(context, forecast), channel_layer_norm(context[:, -1]))


def test_change_cells_ignore_persistent_fire() -> None:
    from fireline.models.jepa import change_cells

    today = torch.zeros(1, 64, 64, dtype=torch.bool)
    tomorrow = torch.zeros_like(today)
    today[0, 0:8, 0:8] = tomorrow[0, 0:8, 0:8] = True  # persistent fire in cell (0, 0)
    assert change_cells(today, tomorrow).sum() == 0
    tomorrow[0, 40, 40] = True  # new fire in cell (5, 5)
    cells = change_cells(today, tomorrow, radius=1)
    assert cells[0, 5, 5] and cells[0, 4, 4] and not cells[0, 0, 0] and cells.sum() == 9


def test_change_mode_starts_at_copy_and_uses_relative_loss() -> None:
    model = SpreadJEPA(history=5, target_mode="change")
    dynamic, static = _batch()
    z = model.latents(dynamic, static)
    torch.testing.assert_close(z["prediction"], z["copy_last"])  # zero change at initialisation
    loss, stats = model.loss(dynamic, static)
    assert abs(stats["prediction"] - 1.0) < 1e-4  # relative loss: equal to copying at the start
    loss.backward()
    assert all(p.grad is not None for p in model.predictor.parameters())


def test_invalid_variant_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        SpreadJEPA(weighting="everything")
