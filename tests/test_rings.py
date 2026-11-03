from __future__ import annotations

import numpy as np
import pytest

from fireline.eval.rings import distance_prior, far_cells


def test_far_cells_scores_and_targets() -> None:
    region = np.zeros((8, 8), dtype=bool)
    region[:, 4:] = True  # far region: right half
    score = np.zeros((8, 8))
    score[0, 0] = 0.9  # outside the region: ignored
    score[1, 5] = 0.4
    score[6, 6] = 0.2
    tomorrow = np.zeros((8, 8), dtype=bool)
    tomorrow[0, 1] = True  # outside the region: not a far target
    tomorrow[7, 7] = True
    scores, targets = far_cells(score, tomorrow, region, cell=4)
    # Cells in row-major order: (0,0) and (1,0) hold no region pixels and are skipped.
    np.testing.assert_allclose(scores, [0.4, 0.2])
    np.testing.assert_array_equal(targets, [False, True])


def test_far_cells_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        far_cells(np.zeros((6, 8)), np.zeros((6, 8), bool), np.ones((6, 8), bool), cell=4)


def test_distance_prior() -> None:
    today = np.zeros((5, 5), dtype=bool)
    today[2, 2] = True
    prior = distance_prior(today)
    assert prior[2, 2] == 1.0 and prior[2, 4] == pytest.approx(1 / (1 + 0.75))
    assert prior[0, 0] < prior[2, 3] < prior[2, 2]
    assert not distance_prior(np.zeros((5, 5), dtype=bool)).any()
