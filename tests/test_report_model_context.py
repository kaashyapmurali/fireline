"""Published context is cited data; seed variability is computed from saved validation-only runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("report_metrics", Path(__file__).parents[1] / "report/scripts/metrics.py")
assert spec is not None and spec.loader is not None
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_model_context_keeps_paper_versions_and_derives_seed_noise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "RUNS", tmp_path)
    for run, results in (
        ("v2-seed-1", {"reference": {"all": {"ap": 0.4}}, "best": {"all": {"ap": 0.5}}}),
        ("v2-seed-2", {"best": {"all": {"ap": 0.6}}}),
    ):
        folder = tmp_path / run
        folder.mkdir()
        (folder / "rings-2020.json").write_text(json.dumps({"results": results}))
    context = metrics.model_context()
    assert context["seed_check"]["n"] == 3 and context["seed_check"]["year"] == 2020
    assert context["seed_check"]["mean_ap"] == pytest.approx(0.5)
    assert context["seed_check"]["sd_ap"] == pytest.approx(0.1)  # sample, not population SD
    comparison = context["attention_comparison"]
    assert comparison["evaluation_dataset"] == "original WSTS"
    assert comparison["gain_min"] == pytest.approx(0.006)
    assert comparison["gain_max"] == pytest.approx(0.015)
    rows = {r["id"]: r for r in context["models"]}
    assert rows["res18-original"]["parameters"] == [14400000, 14700000]
    assert rows["res18-updated"]["parameters"] == [14300000, 14400000]
    assert rows["convlstm"]["parameters"] == [240000]
    assert all("ap" not in row for row in rows.values())  # no scores in the size/design rows
    assert all(row["source"] in context["sources"] for row in rows.values())
    json.dumps(context, allow_nan=False)


def test_constant_brier_is_derived_from_saved_prevalence_and_keeps_model_scores(tmp_path: Path) -> None:
    rows = {name: {"all": {"prevalence": 0.2, "brier": brier, "ap": ap}}
            for name, brier, ap in [("persistence", 0.21, 0.3), ("neighbourhood", 0.12, 0.4), ("best", 0.14, 0.6)]}
    path = tmp_path / "rings.json"
    path.write_text(json.dumps({"samples": 7, "results": rows}))
    report = metrics.rings(path, {"supervised": "best"})
    reference = report["constant_prevalence_reference"]
    assert reference["brier"] == pytest.approx(0.16)
    assert reference["prevalence"] == 0.2 and "hindsight" in reference["note"]
    assert report["arms"]["supervised"]["all"]["brier"] == 0.14
    assert report["arms"]["supervised"]["all"]["ap"] == 0.6
    rows["persistence"]["all"]["prevalence"] = 1.1
    path.write_text(json.dumps({"samples": 7, "results": rows}))
    with pytest.raises(ValueError, match="prevalence"):
        metrics.rings(path, {"supervised": "best"})
