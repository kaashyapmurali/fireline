from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fireline.eval.metrics import PixelMetrics


@pytest.mark.parametrize("binary", [True, False])
def test_average_precision_matches_sklearn(binary: bool) -> None:
    rng = np.random.default_rng(0)
    targets = rng.random(20_000) < 0.02
    scores = (rng.random(20_000) < 0.3) | targets & (rng.random(20_000) < 0.6) if binary else np.clip(
        0.3 * targets + rng.random(20_000) * 0.7, 0, 1)  # fmt: skip
    scores = scores.astype(np.float64)
    metrics = PixelMetrics()
    for chunk in np.array_split(np.arange(20_000), 7):  # streaming in pieces
        metrics.update(scores[chunk], targets[chunk])
    tolerance = 1e-9 if binary else 2e-3  # binned for continuous scores
    assert metrics.compute()["ap"] == pytest.approx(average_precision_score(targets, scores), abs=tolerance)


def test_threshold_metrics_and_valid_mask() -> None:
    scores = np.array([[0.9, 0.6, 0.1], [0.2, 0.7, 0.99]])
    targets = np.array([[1, 0, 1], [0, 1, 0]])
    valid = np.array([[True, True, True], [True, True, False]])
    metrics = PixelMetrics()
    metrics.update(scores, targets, valid)
    result = metrics.compute()
    assert result["pixels"] == 5 and result["positive_pixels"] == 3
    assert result["precision"] == pytest.approx(2 / 3) and result["recall"] == pytest.approx(2 / 3)
    assert result["iou"] == pytest.approx(2 / 4) and result["flagged_over_observed"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx((0.01 + 0.36 + 0.81 + 0.04 + 0.09) / 5)


def test_rejects_non_probabilities() -> None:
    with pytest.raises(ValueError):
        PixelMetrics().update(np.array([1.5]), np.array([1]))
