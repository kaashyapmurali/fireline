from __future__ import annotations

import numpy as np

from fireline.analysis.far_fire import RING_NAMES, explain_far, ring_map


def test_ring_map() -> None:
    today = np.zeros((40, 40), dtype=bool)
    today[20, 20] = True
    rings = ring_map(today)
    assert rings[20, 20] == 0 and rings[20, 22] == 1 and rings[20, 25] == 2 and rings[20, 30] == 3
    assert rings[20, 39] == 4 and rings[0, 0] == 4  # 19-20 px away (square distance)
    assert ring_map(np.zeros((5, 5), dtype=bool)).max() == RING_NAMES.index("no fire today")


def test_explain_far_categories() -> None:
    days, h, w = 6, 40, 40
    mask = np.zeros((days, h, w), dtype=bool)
    t = 4
    mask[t, 20, 5] = True  # today's fire, west side
    mask[t + 1, 20, 30] = True  # far, directly east: downwind of a west wind
    mask[t + 1, 5, 5] = True  # far, burned two days ago: recent fire
    mask[t - 2, 5, 5] = True
    mask[t + 1, 35, 35] = True  # far, unobserved today
    mask[t + 1, 38, 1] = True  # far and upwind (south-west of the fire): nothing explains it
    far = ring_map(mask[t]) >= 3
    observed = np.ones((h, w), dtype=bool)
    observed[35, 35] = False
    wind_u, wind_v = np.full((h, w), 5.0), np.zeros((h, w))  # blowing east
    out = explain_far(mask, t, wind_u, wind_v, observed, far)
    assert out["downwind"][20, 30] and not out["downwind"][5, 5]
    assert out["recent_fire"][5, 5] and out["unobserved_today"][35, 35]
    assert out["isolated"][38, 1] and not out["isolated"][20, 30]


def test_sample_counts() -> None:
    from fireline.eval.rings import sample_counts

    today = np.zeros((30, 30), dtype=bool)
    today[10, 10] = True
    tomorrow = today.copy()
    tomorrow[10, 25] = True  # far fire
    prob = np.zeros((30, 30))
    prob[10, 10] = prob[10, 25] = prob[0, 0] = 0.9  # near hit, far hit, far false alarm
    counts = sample_counts(prob, tomorrow, ring_map(today), np.ones((30, 30), dtype=bool))
    assert counts == {"near_pos": 1, "near_tp": 1, "near_fp": 0, "far_pos": 1, "far_tp": 1, "far_fp": 1}
