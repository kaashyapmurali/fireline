"""Collect measured numbers from saved run outputs, plus cited reference facts from report/sources/model-size.json.

Inputs are the downloaded run outputs under ``outputs/modal/`` (Git-ignored); see ``report/README.md``. The test-year
section is filled from ``outputs/modal/test-2021-01/`` once that single evaluation has run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUNS = Path("outputs/modal")
RINGS = ("on fire today", "1-2 px", "3-5 px", "6-10 px", "11-20 px", ">20 px", "no fire today")
GROUPS = (*RINGS, "near (<= 5 px)", "far (> 5 px or no fire today)", "all")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _curve(path: Path) -> list[dict]:
    """Validation AP per evaluated epoch from a training run's metrics.jsonl."""
    return [{"epoch": r["epoch"], "ap": r["validation"]["ap"]} for r in _jsonl(path) if "validation" in r]


def _best_last(curve: list[dict]) -> dict:
    best = max(curve, key=lambda r: r["ap"])
    return {"best": best["ap"], "best_epoch": best["epoch"], "last": curve[-1]["ap"], "last_epoch": curve[-1]["epoch"]}


def _metrics_row(metrics: dict) -> dict:
    keys = ("ap", "f1", "iou", "precision", "recall", "flagged_over_observed", "brier", "prevalence",
            "pixels", "positive_pixels")  # fmt: skip
    row = {k: metrics.get(k) for k in keys}
    ap, prevalence = row["ap"], row["prevalence"]
    row["lift"] = (ap / prevalence if ap is not None and prevalence is not None
                   and np.isfinite(ap) and np.isfinite(prevalence) and 0 <= ap <= 1 and 0 < prevalence <= 1 else None)
    return row


def per_day(path: Path, min_fire: int = 10) -> dict:
    """Per fire-day scores at threshold 0.5 (supervised model): F1 against the share of tomorrow's fire that is far.

    Only days with at least ``min_fire`` fire pixels tomorrow (F1 on a handful of pixels is noise). Pooled F1 per
    far-share bin uses summed counts, so large days weigh more, as in the overall metric.
    """
    rows = _json(path)["per_sample"]
    points, bins = [], {(0.0, 0.1): [0, 0, 0], (0.1, 0.3): [0, 0, 0], (0.3, 0.6): [0, 0, 0], (0.6, 1.01): [0, 0, 0]}
    for r in rows:
        pos = r["near_pos"] + r["far_pos"]
        if pos < min_fire:
            continue
        tp, fp = r["near_tp"] + r["far_tp"], r["near_fp"] + r["far_fp"]
        far = r["far_pos"] / pos
        points.append([round(far, 3), round(2 * tp / (2 * tp + fp + (pos - tp)), 3), pos])
        for (lo, hi), acc in bins.items():
            if lo <= far < hi:
                acc[0] += tp
                acc[1] += fp
                acc[2] += pos - tp
    f1s = sorted(p[1] for p in points)
    return {
        "min_fire": min_fire,
        "points": points,
        "median_f1": f1s[len(f1s) // 2],
        "bins": [
            {
                "from": lo,
                "to": min(hi, 1.0),
                "days": sum(1 for p in points if lo <= p[0] < hi),
                "f1": 2 * a[0] / (2 * a[0] + a[1] + a[2]) if a[0] + a[1] + a[2] else None,
            }
            for (lo, hi), a in bins.items()
        ],
    }


def concentration(path: Path, steps: int = 60) -> dict:
    """Cumulative share of far fire against the share of fire-days, days sorted by their far fire (largest first)."""
    far = sorted((r["far_pos"] for r in _json(path)["per_sample"]), reverse=True)
    total, days = sum(far) or 1, len(far)
    cumulative = np.cumsum(far) / total
    index = sorted({round(x) for x in np.geomspace(1, days, steps)})
    return {"fire_days": days, "points": [[k / days, float(cumulative[k - 1])] for k in index],
            "days_for_half": int(np.searchsorted(cumulative, 0.5) + 1)}  # fmt: skip


def leak_per_fire(min_pixels: int = 20) -> list[list[float]]:
    """Per fire: median NDVI change at updates for never-burning neighbours and for pixels burning within the window."""
    out = []
    for f in _json(RUNS / "leak-wsts-01" / "composite_leak.json")["per_fire"]:
        c, m = f["counts"], f["median_ndvi_change"]
        if c["burns_in_window"] >= min_pixels and c["never_near"] >= min_pixels:
            out.append([round(m["never_near"], 4), round(m["burns_in_window"], 4), c["burns_in_window"]])
    return out


def dataset() -> dict:
    validation = _json(RUNS / "ingest-wsts-01" / "validation.json")
    years = {int(y): {"fires": v["fires_found"], "images": v["images"], "fire_pixel_fraction": v["fire_pixel_fraction"]}
             for y, v in validation["years"].items()}  # fmt: skip
    return {
        "source": "WildfireSpreadTS (Gerard et al., NeurIPS 2023 Datasets and Benchmarks), CC BY 4.0",
        "channels": 23,
        "resolution_m": 375,
        "years": years,
        "split": {"train": [2018, 2019], "validation": [2020], "test": [2021]},
        "fires_total": sum(v["fires"] for v in years.values()),
        "images_total": sum(v["images"] for v in years.values()),
    }


def rings(path: Path, arms: dict[str, str]) -> dict:
    """AP and threshold metrics per distance ring for the named arms (report name -> name inside the file)."""
    report = _json(path)
    results = report["results"]
    prevalence = float(results["persistence"]["all"]["prevalence"])
    if not np.isfinite(prevalence) or not 0 <= prevalence <= 1:
        raise ValueError(f"Invalid observed-year prevalence in {path}.")
    out = {"samples": report["samples"], "groups": list(GROUPS), "arms": {},
           "constant_prevalence_reference": {"prevalence": prevalence, "brier": prevalence * (1 - prevalence),
               "note": "Constant at the observed evaluation-year fire share: hindsight reference, not a deployable forecast."}}
    for name, key in {"persistence": "persistence", "neighbourhood": "neighbourhood", **arms}.items():
        out["arms"][name] = {g: _metrics_row(results[key][g]) for g in GROUPS if g in results[key]}
    return out


def far_fire(path: Path) -> dict:
    report = _json(path)
    return {k: report[k] for k in ("fire_pixels_tomorrow", "share_by_ring", "far_share", "far_priority_shares",
                                   "far_any_shares", "far_priority_counts_by_ring")}  # fmt: skip


def per_sample_highlights(path: Path, model: str) -> dict:
    """Concentration of far fire in a few fire-days (from per-sample counts)."""
    rows = _json(path)["per_sample"]
    far = sorted((r["far_pos"] for r in rows), reverse=True)
    total = sum(far) or 1
    return {"model": model, "fire_days": len(rows), "fire_days_with_far_fire": sum(1 for v in far if v > 0),
            "share_of_far_fire_in_top_10_days": sum(far[:10]) / total}  # fmt: skip


def jepa() -> dict:
    variants = {
        "jepa_01": ("jepa-01/pretrain/metrics.jsonl", "jepa-01/probe/metrics.jsonl",
                    "Predict tomorrow's latent (absolute loss)"),
        "jepa_02a": ("jepa-02a/pretrain/metrics.jsonl", "jepa-02a/probe/metrics.jsonl",
                     "Same, loss weighted on cells where fire changes"),
        "jepa_02b": ("jepa-02b/pretrain/metrics.jsonl", "jepa-02b-probe-r2/metrics.jsonl",
                     "Predict the change, loss relative to 'no change'"),
    }  # fmt: skip
    out: dict = {"thresholds": {"rank_min_fraction": 0.25, "prediction_over_copy_max": 0.9, "probe_ap_min": 0.43,
                                "noise_ap": 0.03}, "variants": {}}  # fmt: skip
    for name, (pretrain, probe, label) in variants.items():
        gates = [{"epoch": r["epoch"], **{k: r["gates"][k] for k in ("effective_rank", "latent_dim",
                  "prediction_over_copy_near_fire", "copy_error_near_fire", "prediction_error_near_fire")}}
                 for r in _jsonl(RUNS / pretrain) if "gates" in r]  # fmt: skip
        curve = _curve(RUNS / probe)
        out["variants"][name] = {"label": label, "gates": gates, "probe_curve": curve, "probe": _best_last(curve)}
    for name, path in {
        "random_encoder": "probe-random-03/metrics.jsonl",
        "supervised": "unet-sup-02/metrics.jsonl",
    }.items():
        curve = _curve(RUNS / path)
        out[name] = {"probe_curve": curve, "probe": _best_last(curve)}
    return out


def leak() -> dict:
    report = _json(RUNS / "leak-wsts-01" / "composite_leak.json")
    overall = report["overall"]
    labels = {"burns_in_window": "burns 1-15 days after the composite starts",
              "burns_after_window": "burns more than 15 days after",
              "never_near": "never burns, within 1 km", "never_far": "never burns, farther away",
              "already_burned": "already burned", "burns_on_update_day": "burns on the update day"}  # fmt: skip
    return {
        "window_days": report["window_days"],
        "fires": report["fires"],
        "ndvi": {g: {"label": labels[g], **overall["ndvi"]["groups"][g]} for g in labels},
        "auc_ndvi": overall["ndvi"]["auc_larger_drop"],
        "auc_evi2": overall["evi2"]["auc_larger_drop"],
        "auc_ndvi_by_year": {
            y: v["ndvi"]["auc_larger_drop"]["burns_in_window_vs_never_near"] for y, v in report["by_year"].items()
        },
        "fires_with_effect": report["fires_where_burns_in_window_median_drop_exceeds_never_near"],
        "fires_comparable": report["fires_with_burns_in_window_and_near_control"],
        "per_fire": leak_per_fire(),
    }


def model_context() -> dict:
    """Published sizes stay in one cited source file; seed noise comes from existing validation results, never a rerun."""
    context = _json(Path(__file__).resolve().parents[1] / "sources/model-size.json")
    comparison = context["attention_comparison"]
    gains = np.array(comparison["five_day_utae_res18_ap"]) - np.array(comparison["five_day_stacked_res18_ap"])
    comparison["gain_min"], comparison["gain_max"] = float(gains.min()), float(gains.max())
    first = _json(RUNS / "v2-seed-1/rings-2020.json")
    second = _json(RUNS / "v2-seed-2/rings-2020.json")
    aps = [first["results"]["reference"]["all"]["ap"], first["results"]["best"]["all"]["ap"],
           second["results"]["best"]["all"]["ap"]]
    context["seed_check"] = {"year": 2020, "n": len(aps), "aps": aps, "mean_ap": float(np.mean(aps)),
                             "sd_ap": float(np.std(aps, ddof=1)), "sources": ["v2-seed-1/rings-2020.json",
                             "v2-seed-2/rings-2020.json"], "note": "Three supervised baseline seeds, not JEPA seed noise."}
    return context


def model() -> dict:
    """Architecture facts for the report, from the run config and the model code."""
    from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES
    from fireline.models.spread import SpreadModel

    config = _json(RUNS / "unet-sup-02" / "config.json")
    net = SpreadModel(history=config["history"])
    count = lambda m: sum(p.numel() for p in m.parameters())
    probe = _json(RUNS / "jepa-02b-probe-r2" / "config.json")
    return {
        "parameters": config["parameters"],
        "encoder_parameters": count(net.encoder),
        "context": model_context(),
        "probe_trainable_parameters": probe.get("trainable_parameters"),
        "widths": list(net.encoder.widths),
        "history_days": config["history"],
        "train_crop_px": config["crop"],
        "epochs": config["epochs"],
        "dynamic_features": list(DYNAMIC_FEATURES),
        "static_features": [f for f in STATIC_FEATURES if not f.startswith("landcover_")] + ["landcover (17 classes)"],
        "loss": "binary cross-entropy + soft Dice",
        "optimizer": f"AdamW, learning rate {config['lr']:g}, weight decay {config['weight_decay']:g}",
        "schedule": f"{config['warmup_epochs']:g} warm-up epoch, then cosine decay; gradient norm clipped at 1",
        "batch_size": config["batch_size"],
    }


def wind() -> dict:
    curve, base = _curve(RUNS / "unet-wind-01" / "metrics.jsonl"), _curve(RUNS / "unet-sup-02" / "metrics.jsonl")
    ring = rings(RUNS / "unet-wind-01" / "rings-2020.json",
                 {"wind_best": "wind_best", "supervised_best": "sup_best"})  # fmt: skip
    return {
        "curve": curve,
        "supervised_curve": base,
        "validation_rings": ring,
        "criteria": {"rings": ["3-5 px", "6-10 px"], "min_relative_gain": 0.20, "overall_guard": 0.01},
    }


def baselines_validation() -> dict:
    report = _json(RUNS / "baselines-01" / "persistence-2020.json")
    return {name: _metrics_row(report[name]) for name in ("persistence", "neighbourhood")}


TEST_ARMS = {"supervised": "supervised", "wind": "wind", "jepa_b": "jepa_b", "jepa_01": "jepa_01",
             "random_encoder": "random_encoder"}  # fmt: skip


def collect(require_test: bool = True) -> dict:
    test_dir = RUNS / "test-2021-01"
    has_test = (test_dir / "rings-2021.json").exists()
    if require_test and not has_test:
        raise FileNotFoundError(f"{test_dir}/rings-2021.json missing; run the test-year evaluation first.")
    site = {
        "dataset": dataset(),
        "validation": {
            "baselines": baselines_validation(),
            "supervised_curve": _curve(RUNS / "unet-sup-02" / "metrics.jsonl"),
            "rings": rings(RUNS / "far-fire-01" / "rings-2020.json", {"supervised": "best"}),
            "far_fire": far_fire(RUNS / "far-fire-01" / "far-fire-2020.json"),
            "far_concentration": per_sample_highlights(RUNS / "far-fire-01" / "rings-2020.json", "best"),
            "concentration": concentration(RUNS / "far-fire-01" / "rings-2020.json"),
        },
        "model": model(),
        "jepa": jepa(),
        "leak": leak(),
        "wind": wind(),
        "test": None,
    }
    if has_test:
        site["test"] = {
            "year": 2021,
            "rings": rings(test_dir / "rings-2021.json", TEST_ARMS),
            "far_fire": far_fire(test_dir / "far-fire-2021.json"),
            "far_concentration": per_sample_highlights(test_dir / "rings-2021.json", "supervised"),
            "per_day": per_day(test_dir / "rings-2021.json"),
            "concentration": concentration(test_dir / "rings-2021.json"),
        }
    return site
