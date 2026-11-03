from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_app


def test_smoke_stage_converts_reference_fire_into_its_own_folder() -> None:
    (stage,) = modal_app.ingest_smoke_stages("ingest-smoke-01")
    assert stage[:4] == ["python", "-u", "-m", "fireline.data.ingest"]
    assert stage[stage.index("--fire") + 1] == modal_app.SMOKE_REFERENCE_FIRE
    assert stage[stage.index("--out") + 1] == "/data/smoke/ingest-smoke-01/wsts"
    assert stage[stage.index("--report") + 1] == "/data/smoke/ingest-smoke-01/report.json"


@pytest.mark.parametrize("name", ["", "jobs", "a/b", "../x", "smoke"])
def test_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        modal_app.validate_name(name)


def test_runner_completes_and_refuses_rerun(tmp_path: Path) -> None:
    stages = [[sys.executable, "-c", "print('hello')"]]
    status = modal_app.run_stages("job-a", stages, 30, jobs=tmp_path)
    assert status["state"] == "complete"
    assert "hello" in (tmp_path / "job-a.log").read_text()
    assert json.loads((tmp_path / "job-a.status.json").read_text())["state"] == "complete"
    with pytest.raises(FileExistsError):
        modal_app.run_stages("job-a", stages, 30, jobs=tmp_path)


def test_runner_kills_stage_over_budget(tmp_path: Path) -> None:
    stages = [[sys.executable, "-c", "import time; time.sleep(30)"]]
    with pytest.raises(TimeoutError):
        modal_app.run_stages("job-b", stages, 1, jobs=tmp_path)
    assert json.loads((tmp_path / "job-b.status.json").read_text())["state"] == "failed"


def test_runner_reports_failure_without_retry(tmp_path: Path) -> None:
    stages = [[sys.executable, "-c", "raise SystemExit(3)"], [sys.executable, "-c", "print('never')"]]
    with pytest.raises(RuntimeError, match="status 3"):
        modal_app.run_stages("job-c", stages, 30, jobs=tmp_path)
    assert "never" not in (tmp_path / "job-c.log").read_text()


def test_function_timeout_is_budget_plus_margin() -> None:
    assert modal_app.function_timeout(900) == 1200
    with pytest.raises(ValueError):
        modal_app.function_timeout(modal_app.MAX_SECONDS + 1)


def test_full_ingest_converts_each_year_resumably_then_validates() -> None:
    stages = modal_app.ingest_full_stages("ingest-wsts-01")
    assert len(stages) == 5
    for year, stage in zip(modal_app.WSTS_YEARS, stages[:4], strict=True):
        assert stage[stage.index("--year") + 1] == str(year)
        assert "--all" in stage and "--skip-existing" in stage
        assert stage[stage.index("--out") + 1] == "/data/processed/wsts"
        assert stage[stage.index("--report") + 1] == f"/data/reports/ingest-wsts-01/ingest-{year}.json"
    validate = stages[4]
    assert validate[3] == "fireline.data.validate"
    assert validate[validate.index("--years") + 1 : validate.index("--report")] == ["2018", "2019", "2020", "2021"]


def test_composite_leak_stage_reads_processed_data_and_writes_one_report() -> None:
    (stage,) = modal_app.composite_leak_stages("leak-wsts-01")
    assert stage[3] == "fireline.analysis.composite_leak"
    assert stage[stage.index("--data") + 1] == "/data/processed/wsts"
    assert stage[stage.index("--report") + 1] == "/data/reports/leak-wsts-01/composite_leak.json"


def test_submit_refuses_other_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    with pytest.raises(RuntimeError, match="-e fireline"):
        modal_app.submit("x-01", [["true"]], 60, confirm_run=False)


def test_baselines_never_touch_the_test_year() -> None:
    stages = modal_app.baselines_stages("baselines-01")
    assert [s[4] for s in stages] == ["stats", "evaluate", "evaluate"]
    assert all("2021" not in s for s in stages)
    assert stages[0][stages[0].index("--years") + 1 : stages[0].index("--output")] == ["2018", "2019"]
    assert stages[1][stages[1].index("--years") + 1 : stages[1].index("--report")] == ["2020"]


def test_supervised_training_stage() -> None:
    (stage,) = modal_app.train_supervised_stages("unet-smoke-01", epochs=1)
    assert stage[3] == "fireline.train.supervised"
    assert stage[stage.index("--stats") + 1] == "/data/reports/baselines-01/stats-2018-2019.json"
    assert stage[stage.index("--output") + 1] == "/data/runs/unet-smoke-01"
    assert stage[stage.index("--epochs") + 1] == "1" and "--amp" in stage
    assert "--val-years" not in stage  # defaults: train 2018-2019, validate 2020; the test year is never used


def test_gpu_runs_keep_their_declared_resources() -> None:
    gpu = modal_app.plan_resources(modal_app.run_gpu, cpu_cores=2, memory_mib=4096)
    assert gpu == {"gpu": ["L4", "A10"], "cpu_cores": 4, "memory_mib": 16384}
    cpu = modal_app.plan_resources(None, cpu_cores=4, memory_mib=12288)
    assert cpu == {"gpu": None, "cpu_cores": 4, "memory_mib": 12288}


def test_submit_spawns_gpu_runs_without_overriding_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {}

    class FakeCall:
        object_id = "fc-test"

    class FakeFunction:
        def spawn(self, *args):
            calls["spawn"] = args
            return FakeCall()

    def fake_with_options(**options):
        calls["options"] = options
        return FakeFunction()

    monkeypatch.setenv("MODAL_ENVIRONMENT", "fireline")
    monkeypatch.setattr(modal_app, "ROOT", tmp_path)
    monkeypatch.setattr(modal_app.run_gpu, "with_options", fake_with_options)
    modal_app.submit("gpu-test-01", [["true"]], 600, confirm_run=True, target=modal_app.run_gpu)
    assert calls["options"] == {"timeout": 900}
    record = json.loads((tmp_path / "outputs/modal/gpu-test-01.json").read_text())
    assert record["gpu"] == ["L4", "A10"] and record["cpu_cores"] == 4 and record["call_id"] == "fc-test"


def test_distance_stages_skip_the_test_year() -> None:
    stages = modal_app.distance_stages("distance-01")
    assert [s[4] for s in stages] == ["distance", "distance"] and all("2021" not in s for s in stages)


def test_jepa_stages_pretrain_then_probe_with_frozen_encoder() -> None:
    pretrain, probe = modal_app.train_jepa_stages("jepa-01")
    assert (
        pretrain[3] == "fireline.train.jepa"
        and pretrain[pretrain.index("--output") + 1] == "/data/runs/jepa-01-pretrain"
    )
    assert probe[3] == "fireline.train.supervised" and "--freeze-encoder" in probe
    assert probe[probe.index("--init-encoder") + 1] == "/data/runs/jepa-01-pretrain/encoder.pt"
    supervised = modal_app.train_supervised_stages("x", epochs=30, val_every=2)[0]
    for flag in ("--batch-size", "--epochs", "--val-every", "--seed", "--stats"):  # same recipe as unet-sup-02
        assert probe[probe.index(flag) + 1] == supervised[supervised.index(flag) + 1]


def test_random_frozen_encoder_control_matches_the_probe_recipe() -> None:
    (control,) = modal_app.train_supervised_stages(
        "probe-random-01", epochs=30, val_every=2, random_frozen_encoder=True
    )
    _, probe = modal_app.train_jepa_stages("jepa-01")
    assert "--freeze-encoder" in control and "--random-encoder" in control and "--init-encoder" not in control
    for flag in ("--batch-size", "--epochs", "--val-every", "--seed", "--stats", "--workers"):
        assert control[control.index(flag) + 1] == probe[probe.index(flag) + 1]


def test_jepa_variants_pass_their_switches() -> None:
    (pretrain, _) = modal_app.train_jepa_stages("jepa-02b", target_mode="change")
    assert pretrain[pretrain.index("--target-mode") + 1] == "change"
    assert pretrain[pretrain.index("--weighting") + 1] == "near_fire"
    (pretrain, _) = modal_app.train_jepa_stages("jepa-02a", weighting="change")
    assert pretrain[pretrain.index("--weighting") + 1] == "change"
    assert pretrain[pretrain.index("--target-mode") + 1] == "state"


def test_far_fire_stages_use_validation_only() -> None:
    diagnose, rings = modal_app.far_fire_stages("far-fire-01")
    assert diagnose[3] == "fireline.analysis.far_fire" and rings[3] == "fireline.eval.rings"
    assert all("2021" not in s for s in (diagnose, rings))
    assert "best=/data/runs/unet-sup-02/best.pt" in rings and rings[rings.index("--samples-for") + 1] == "best"


def test_wind_run_matches_the_supervised_recipe_plus_features() -> None:
    train, rings = modal_app.train_wind_stages("unet-wind-01")
    (base,) = modal_app.train_supervised_stages("unet-wind-01", epochs=30, val_every=2)
    assert train == [*base, "--wind-features"]
    assert "sup_best=/data/runs/unet-sup-02/best.pt" in rings and "wind_best=/data/runs/unet-wind-01/best.pt" in rings
    assert all("2021" not in s for s in (train, rings))


def test_probe_rerun_matches_the_jepa_probe_except_paths() -> None:
    (rerun,) = modal_app.train_probe_stages("jepa-02b-probe-r2", encoder_run="jepa-02b-pretrain")
    _, original = modal_app.train_jepa_stages("jepa-02b", target_mode="change")
    assert rerun[rerun.index("--init-encoder") + 1] == "/data/runs/jepa-02b-pretrain/encoder.pt"
    assert rerun[rerun.index("--output") + 1] == "/data/runs/jepa-02b-probe-r2"
    strip = lambda a: [x for i, x in enumerate(a) if i not in (a.index("--output") + 1,)]
    assert strip(rerun) == strip(original)


def test_test_year_stage_scores_every_arm_on_2021_only() -> None:
    rings, diagnose = modal_app.test_year_stages("test-2021-01")
    years = rings[rings.index("--years") + 1 : rings.index("--checkpoint")]
    assert years == ["2021"] and diagnose[diagnose.index("--years") + 1] == "2021"
    for name, run in modal_app.TEST_ARMS.items():
        assert f"{name}=/data/runs/{run}/best.pt" in rings
    assert rings[rings.index("--stats") + 1].endswith("stats-2018-2019.json")  # training-year statistics


def test_only_the_test_year_stage_touches_2021() -> None:
    builders = [modal_app.baselines_stages("x"), modal_app.distance_stages("x"), modal_app.far_fire_stages("x"),
                modal_app.train_wind_stages("x"), modal_app.train_jepa_stages("x"),
                modal_app.train_supervised_stages("x", epochs=1),
                *(modal_app.train_v2_stages("x", variant=v) for v in modal_app.V2_VARIANTS)]  # fmt: skip
    assert all("2021" not in arg for stages in builders for stage in stages for arg in stage)


def test_v2_runs_are_the_supervised_recipe_plus_variant_options() -> None:
    (base,) = modal_app.train_supervised_stages("v2-seed-1", epochs=30, val_every=2, seed=1)
    train, rings = modal_app.train_v2_stages("v2-seed-1", variant="base", seed=1, reference_run="unet-sup-02")
    assert train == base
    assert "reference=/data/runs/unet-sup-02/best.pt" in rings and "best=/data/runs/v2-seed-1/best.pt" in rings
    assert rings[rings.index("--cells") + 1 : rings.index("--cells") + 3] == ["8", "4"]
    far, _ = modal_app.train_v2_stages("v2-far-01", variant="far")
    history, history_rings = modal_app.train_v2_stages("v2-history-01", variant="history")
    assert far[-6:] == ["--far-weight", "5", "--far-crop-prob", "0.5", "--far-oversample", "3"]
    assert history[-1] == "--history-channels" and "reference" not in " ".join(history_rings)
    with pytest.raises(ValueError):
        modal_app.train_v2_stages("x", variant="unknown")
