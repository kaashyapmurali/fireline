"""Training/evaluation samples: which days form a sample, fire-biased crops and exact geometric augmentation.

A sample uses input days ``t - history + 1 .. t`` and the active-fire mask of day ``t + 1`` as target. Every model
is evaluated on the same targets: those with at least ``min_history`` earlier days (default 5), whatever history
length the model itself uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES, STATIC_VECTOR_PAIRS, VECTOR_PAIRS

FIRE_INDEX = DYNAMIC_FEATURES.index("active_fire")


def target_days(n_days: int, history: int, min_history: int = 5) -> list[int]:
    """Indices ``t`` of the last input day of every sample in a fire with ``n_days`` days."""
    first = max(history, min_history) - 1
    return list(range(first, n_days - 1))


@dataclass(frozen=True)
class Sample:
    dynamic: np.ndarray  # (T, C_dynamic, H, W)
    static: np.ndarray  # (C_static, H, W)
    target: np.ndarray  # (H, W) uint8


def choose_crop(
    input_fire: np.ndarray, target: np.ndarray, size: int, rng: np.random.Generator, tries: int = 10
) -> tuple[slice, slice]:
    """Random square window preferring fire in the target, then in the inputs (the WSTS training heuristic).

    ``input_fire`` (T, H, W) and ``target`` (H, W) are binary masks; only they are needed to choose the window.
    """
    height, width = target.shape
    if height < size or width < size:
        raise ValueError(f"Crop {size} larger than image {height}x{width}.")
    best, best_score = (0, 0), -1.0
    for _ in range(tries):
        top, left = int(rng.integers(0, height - size + 1)), int(rng.integers(0, width - size + 1))
        window = (slice(top, top + size), slice(left, left + size))
        score = 1000 * target[window].mean() + input_fire[(slice(None), *window)].mean()
        if score > best_score:
            best, best_score = (top, left), score
    return slice(best[0], best[0] + size), slice(best[1], best[1] + size)


def fire_biased_crop(sample: Sample, size: int, rng: np.random.Generator, tries: int = 10) -> Sample:
    window = choose_crop(sample.dynamic[:, FIRE_INDEX], sample.target, size, rng, tries)
    return Sample(sample.dynamic[(..., *window)], sample.static[(..., *window)], sample.target[window])


def _vector_indices(names: tuple[str, ...], pairs: tuple[tuple[str, str], ...]) -> list[tuple[int, int]]:
    return [(names.index(a), names.index(b)) for a, b in pairs]


_DYNAMIC_VECTORS = _vector_indices(DYNAMIC_FEATURES, VECTOR_PAIRS)
_STATIC_VECTORS = _vector_indices(STATIC_FEATURES, STATIC_VECTOR_PAIRS)


def transform(sample: Sample, hflip: bool, vflip: bool, rotations: int) -> Sample:
    """Flip/rotate images and rotate the (east, north) vector features to match. Row 0 is north (north-up rasters).

    hflip mirrors east-west (east -> -east); vflip mirrors north-south (north -> -north); each 90-degree
    counter-clockwise rotation maps (east, north) to (-north, east).
    """
    dynamic, static, target = sample.dynamic.copy(), sample.static.copy(), sample.target.copy()
    for array, pairs, channel_axis in ((dynamic, _DYNAMIC_VECTORS, 1), (static, _STATIC_VECTORS, 0)):
        for e, n in pairs:
            east, north = np.take(array, e, axis=channel_axis), np.take(array, n, axis=channel_axis)
            if hflip:
                east = -east
            if vflip:
                north = -north
            for _ in range(rotations % 4):
                east, north = -north, east
            index_e = [slice(None)] * array.ndim
            index_e[channel_axis] = e
            index_n = list(index_e)
            index_n[channel_axis] = n
            array[tuple(index_e)], array[tuple(index_n)] = east, north

    def geometric(x: np.ndarray) -> np.ndarray:
        if hflip:
            x = x[..., :, ::-1]
        if vflip:
            x = x[..., ::-1, :]
        return np.rot90(x, k=rotations % 4, axes=(-2, -1))

    return Sample(*(np.ascontiguousarray(geometric(x)) for x in (dynamic, static, target)))


def random_transform(sample: Sample, rng: np.random.Generator) -> Sample:
    return transform(sample, bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(4)))


def pad_to_multiple(sample: Sample, multiple: int = 32, mode: str = "reflect") -> tuple[Sample, np.ndarray]:
    """Pad bottom/right to a multiple of ``multiple``; returns the padded sample and a valid-pixel mask.

    Inputs are padded by reflection by default: zero padding creates flat strips that training crops never contain,
    and a trained model produced false alarms along them (unet-sup-01 diagnosis). The target is zero-padded and the
    padded pixels are excluded from metrics through the valid mask.
    """
    height, width = sample.target.shape
    pad_h, pad_w = -height % multiple, -width % multiple
    valid = np.zeros((height + pad_h, width + pad_w), dtype=bool)
    valid[:height, :width] = True

    def pad(x: np.ndarray, pad_mode: str) -> np.ndarray:
        return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, pad_h), (0, pad_w)], mode=pad_mode)  # type: ignore[call-overload]

    padded = Sample(pad(sample.dynamic, mode), pad(sample.static, mode), pad(sample.target, "constant"))
    return padded, valid
