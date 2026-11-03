"""Pixel metrics split by distance from today's fire, for saved models and the training-free baselines.

    python -m fireline.eval.rings --data DIR --stats STATS.json --cache-dir CACHE --years 2020 \
        --checkpoint best=RUN/best.pt --checkpoint last=RUN/last.pt --report OUT.json

Each validation target is scored on full images (reflection padding, padded pixels excluded), and every pixel is
assigned the distance ring of ``fireline.analysis.far_fire.ring_map``. AP within a ring answers "how well are fire
pixels ranked among the pixels at that distance"; the "far" group pools rings beyond 5 px and no-fire-today images.

``--cells 8 4`` adds a coarse far-fire score: the far region is cut into square cells of that many pixels (8 px = 3 km),
a cell is positive if any of its far pixels burns tomorrow and is scored by its highest far-pixel score. It asks
whether far fire is placed in the right area rather than the right pixel. A distance prior (1 / (1 + km to today's
fire), 0 with no fire today) is scored alongside as the reference for "closer is likelier".
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from fireline.analysis.far_fire import RING_NAMES, ring_map
from fireline.baselines.persistence import box_mean, fire_paths
from fireline.data.cache import materialize_all
from fireline.data.history_features import HISTORY_FEATURES
from fireline.data.inputs import DYNAMIC_FEATURES, Stats
from fireline.data.wind_features import WIND_FEATURES
from fireline.eval.fire_batches import iterate_fires, predict_fire
from fireline.eval.metrics import PixelMetrics
from fireline.models.spread import SpreadModel

GROUPS = (*RING_NAMES, "near (<= 5 px)", "far (> 5 px or no fire today)", "all")
PIXEL_KM = 0.375
DISTANCE_PRIOR = "distance prior"


def distance_prior(today: np.ndarray) -> np.ndarray:
    """1 / (1 + Euclidean km to today's nearest fire pixel); zero everywhere when there is no fire today."""
    if not today.any():
        return np.zeros(today.shape)
    return 1.0 / (1.0 + PIXEL_KM * distance_transform_edt(~today))


def far_cells(score: np.ndarray, tomorrow: np.ndarray, region: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores and targets of the ``cell`` x ``cell`` cells that contain at least one ``region`` pixel.

    A cell's score is its highest score over region pixels; its target is whether any region pixel burns tomorrow.
    Image sides must be multiples of ``cell`` (evaluation pads to multiples of 32).
    """
    h, w = region.shape
    if h % cell or w % cell:
        raise ValueError(f"Image {h}x{w} is not a multiple of the cell size {cell}.")

    def blocks(a: np.ndarray) -> np.ndarray:
        return a.reshape(h // cell, cell, w // cell, cell)

    eligible = blocks(region).any(axis=(1, 3))
    best = blocks(np.where(region, score, -1.0)).max(axis=(1, 3))
    burns = blocks(tomorrow & region).any(axis=(1, 3))
    return best[eligible], burns[eligible]


def _group_masks(rings: np.ndarray) -> dict[str, np.ndarray]:
    masks = {name: rings == i for i, name in enumerate(RING_NAMES)}
    masks["near (<= 5 px)"] = rings <= 2
    masks["far (> 5 px or no fire today)"] = rings >= 3
    masks["all"] = np.ones(rings.shape, dtype=bool)
    return masks


def sample_counts(prob: np.ndarray, tomorrow: np.ndarray, rings: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    """Per-sample counts at threshold 0.5, near (<= 5 px) and far (> 5 px or no fire today)."""
    flagged = (prob >= 0.5) & valid
    out = {}
    for name, region in (("near", (rings <= 2) & valid), ("far", (rings >= 3) & valid)):
        out[f"{name}_pos"] = int((tomorrow & region).sum())
        out[f"{name}_tp"] = int((flagged & tomorrow & region).sum())
        out[f"{name}_fp"] = int((flagged & ~tomorrow & region).sum())
    return out


def evaluate(models: dict[str, SpreadModel], fire_dirs: list[Path], stats: Stats, device: torch.device,
             workers: int = 4, batch_size: int = 4, amp: bool = False, sample_model: str | None = None,
             cells: tuple[int, ...] = (), history_channels: bool = False) -> dict:  # fmt: skip
    scorers = ["persistence", "neighbourhood", *models]
    metrics = {s: {g: PixelMetrics() for g in GROUPS} for s in scorers}
    cell_metrics = {c: {s: PixelMetrics() for s in [*scorers, DISTANCE_PRIOR]} for c in cells}
    history = next(iter(models.values())).history if models else 5
    samples = 0
    per_sample: list[dict] = []
    wind = any(m.extra_channels for m in models.values()) and not history_channels
    for fire in iterate_fires(fire_dirs, stats, history, workers, wind=wind, history_channels=history_channels):
        valid = fire.valid.numpy()
        mask = fire.fire_mask.numpy()
        predictions: dict[str, dict[int, np.ndarray]] = {name: {} for name in models}
        for name, model in models.items():
            channels = len(DYNAMIC_FEATURES) + model.extra_channels  # models without wind features see the base set
            view = replace(fire, dynamic=fire.dynamic[:, :channels])
            for ts, prob in predict_fire(model, view, device, amp, batch_size):
                for t, p in zip(ts, prob.cpu().numpy(), strict=True):
                    predictions[name][t] = p
        for t in fire.targets:
            today, tomorrow = mask[t], mask[t + 1]
            groups = {g: m & valid for g, m in _group_masks(ring_map(today)).items()}
            scores = {"persistence": today.astype(np.float64), "neighbourhood": box_mean(today, 2)}
            scores.update({name: predictions[name][t] for name in models})
            for scorer, score in scores.items():
                for group, selection in groups.items():
                    if selection.any():
                        metrics[scorer][group].update(score, tomorrow, selection)
            if cells:
                far = groups["far (> 5 px or no fire today)"]
                cell_scores = {**scores, DISTANCE_PRIOR: distance_prior(today)}
                for cell in cells:
                    for scorer, score in cell_scores.items():
                        cell_metrics[cell][scorer].update(*far_cells(score, tomorrow, far, cell))
            if sample_model is not None:
                counts = sample_counts(scores[sample_model], tomorrow, ring_map(today), valid)
                per_sample.append({"fire": fire.fire, "t": t, **counts})
            samples += 1
    return {
        "samples": samples,
        "results": {s: {g: m.compute() for g, m in groups_.items()} for s, groups_ in metrics.items()},
        "per_sample": per_sample,
        "cells": {f"{c} px": {s: m.compute() for s, m in by_scorer.items()} for c, by_scorer in cell_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2020])
    parser.add_argument("--checkpoint", action="append", default=[], help="name=path to a SpreadModel checkpoint.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-fires", type=int, default=0, help="Development only.")
    parser.add_argument("--samples-for", help="Record per-sample near/far counts for this checkpoint name.")
    parser.add_argument("--cells", type=int, nargs="*", default=[], help="Coarse far-fire cell sizes in px, e.g. 8 4.")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"{args.report} exists; refusing to overwrite.")
    device = torch.device(args.device)
    models, kinds = {}, set()
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        checkpoint = torch.load(path, map_location="cpu")
        wind = bool(checkpoint["config"].get("wind_features", False))
        past = bool(checkpoint["config"].get("history_channels", False))
        kinds |= {"wind"} if wind else {"history"} if past else set()
        model = SpreadModel(history=int(checkpoint["config"].get("history", 5)),
                            extra_channels=len(WIND_FEATURES) if wind else len(HISTORY_FEATURES) if past else 0)  # fmt: skip
        model.load_state_dict(checkpoint["model"])
        models[name] = model.to(device).eval()
    if len(kinds) > 1:
        raise ValueError("Checkpoints with wind features and with history channels need separate evaluations.")
    start = time.perf_counter()
    fire_dirs = materialize_all(
        fire_paths(args.data, args.years)[: args.max_fires or None], args.cache_dir, max(args.workers, 1)
    )
    result = evaluate(models, fire_dirs, Stats.from_json(args.stats), device, args.workers, amp=args.amp,
                      sample_model=args.samples_for, cells=tuple(args.cells),
                      history_channels="history" in kinds)  # fmt: skip
    report = {"years": args.years, "checkpoints": args.checkpoint, "rings": list(RING_NAMES), **result,
              "seconds": round(time.perf_counter() - start, 1)}  # fmt: skip
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for scorer, groups in report["results"].items():
        print(scorer, {g: (round(v["ap"], 4) if v["ap"] is not None else None) for g, v in groups.items()}, flush=True)
    for cell, by_scorer in report["cells"].items():
        print("cells", cell, {s: (round(v["ap"], 4) if v["ap"] is not None else None) for s, v in by_scorer.items()})


if __name__ == "__main__":
    main()
