"""Run Fireline jobs on Modal: data preparation and analyses on CPU, training and evaluation on GPU.

    modal run -e fireline --detach modal_app.py::composite_leak --run-name leak-wsts-02 --confirm-run
    modal run -e fireline --detach modal_app.py::baselines --run-name baselines-01 --confirm-run
    modal run -e fireline --detach modal_app.py::train_supervised --run-name unet-smoke-01 --epochs 1 --confirm-run
    modal run -e fireline --detach modal_app.py::distance --run-name distance-01 --confirm-run
    modal run -e fireline --detach modal_app.py::train_jepa --run-name jepa-01 --confirm-run
    modal run -e fireline --detach modal_app.py::far_fire --run-name far-fire-01 --confirm-run
    modal run -e fireline --detach modal_app.py::train_wind --run-name unet-wind-01 --confirm-run
    modal run -e fireline --detach modal_app.py::test_year --run-name test-2021-01 --confirm-run   # scored ONCE
    modal run -e fireline --detach modal_app.py::train_probe --run-name jepa-02b-probe-r2 --encoder-run jepa-02b-pretrain --confirm-run
    modal run -e fireline --detach modal_app.py::train_v2 --run-name v2-seed-1 --variant base --seed 1 --confirm-run

All Fireline apps and volumes live in the Modal environment `fireline` (not the shared default `main`). Always pass
`-e fireline`; entrypoints refuse to submit from another environment. (The first three runs, ingest-smoke-01,
ingest-wsts-01 and leak-wsts-01, ran in `main` before this environment existed; migrate-01 copied their data here,
using a one-off migration function that has since been removed.)

Without --confirm-run the entrypoint only prints the plan (note: `modal run` still starts the app and image).

Always use `modal run --detach`: without it Modal stops the app (cancelling the run) when the local entrypoint
returns. `--confirm-run` is required to submit.

Volume `fireline-data`: converted data and job records. Layout:
    /data/jobs/<run>.status.json, /data/jobs/<run>.log    job status and full log
    /data/smoke/<run>/...                                  smoke-test outputs (disposable)
    /data/processed/wsts/<year>/<fire>.npz                 converted WSTS (one file per fire)
    /data/reports/<run>/...                                reports (ingest, validation, baselines, statistics)
    /data/runs/<run>/...                                   training runs: config, metrics.jsonl, best.pt, last.pt

The runner never overwrites an existing job, enforces a work-time budget by killing the stage's process group,
commits the volume while running and at the end, and never retries. Hard caps: `work_seconds` (stage killed after
it), a per-run Modal function timeout of `work_seconds` + 300 s (set at submission with `with_options`), and fixed
resources (2 cores, 4 GiB).
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA = Path("/data")
JOBS = DATA / "jobs"
MAX_SECONDS = 4 * 3600
CPU_CORES, MEMORY_MIB = 2, 4096
RESERVED = ("jobs", "smoke", "processed", "reports", "runs")
STATS = DATA / "reports" / "baselines-01" / "stats-2018-2019.json"
GPU: list[str] = ["L4", "A10"]  # fallback order: Modal uses the first type with free capacity
GPU_CPU_CORES, GPU_MEMORY_MIB = 4, 16384  # 16 GiB (32 GiB did not speed up validation)
# No explicit ephemeral_disk: Modal only accepts requests of 512 GiB to 3 TiB. The training script checks free disk
# for its uncompressed fire cache (about 28 GB) before building it and fails fast if the default disk is too small.
WSTS_YEARS = (2018, 2019, 2020, 2021)

app = modal.App("fireline", include_source=False)
image = (
    modal.Image.debian_slim(python_version="3.14.8")
    .uv_pip_install("numpy==2.5.4", "rasterio==1.5.2")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("fireline")
    .add_local_file(ROOT / "modal_app.py", "/root/modal_app.py")
)
ENVIRONMENT = "fireline"
gpu_image = (
    modal.Image.debian_slim(python_version="3.14.8")
    .uv_pip_install("numpy==2.5.4", "rasterio==1.5.2", "torch==2.15.0", "scipy==1.18.1")
    .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"})
    .add_local_python_source("fireline")
    .add_local_file(ROOT / "modal_app.py", "/root/modal_app.py")
)
data_volume = modal.Volume.from_name("fireline-data", environment_name=ENVIRONMENT, create_if_missing=True, version=1)


def validate_name(name: str) -> str:
    if not name or not name.isascii() or not all(c.isalnum() or c in "-_" for c in name) or name in RESERVED:
        raise ValueError("Use a non-reserved run name made of letters, numbers, '-' and '_'.")
    return name


def module(name: str, *args: str) -> list[str]:
    return ["python", "-u", "-m", f"fireline.{name}", *args]


# ---------------------------------------------------------------- command builders (pure; tested)

SMOKE_REFERENCE_FIRE = "fire_23160475"  # converted locally in Step 1; compared byte-for-byte after the smoke run


def ingest_smoke_stages(run_name: str, *, year: int = 2019, first: int = 15, workers: int = 8) -> list[list[str]]:
    """Convert the Step 1 reference fire plus the first `first` other fires of `year` with `workers` connections."""
    out = DATA / "smoke" / validate_name(run_name)
    return [
        module(
            "data.ingest", "--year", str(year), "--fire", SMOKE_REFERENCE_FIRE, "--first", str(first),
            "--workers", str(workers), "--out", str(out / "wsts"), "--report", str(out / "report.json"),
        )
    ]  # fmt: skip


def ingest_full_stages(run_name: str, *, years: tuple[int, ...] = WSTS_YEARS, workers: int = 12) -> list[list[str]]:
    """Convert every fire of each year (resumable: existing fires are skipped), then validate all years."""
    reports = DATA / "reports" / validate_name(run_name)
    out = DATA / "processed" / "wsts"
    stages = [
        module(
            "data.ingest", "--year", str(year), "--all", "--skip-existing", "--workers", str(workers),
            "--out", str(out), "--report", str(reports / f"ingest-{year}.json"),
        )
        for year in years
    ]  # fmt: skip
    stages.append(
        module(
            "data.validate",
            "--out",
            str(out),
            "--years",
            *map(str, years),
            "--report",
            str(reports / "validation.json"),
        )
    )
    return stages


def composite_leak_stages(run_name: str, *, years: tuple[int, ...] = WSTS_YEARS, workers: int = 2) -> list[list[str]]:
    """Composite NDVI/EVI2 future-information analysis over all converted fires (read-only on the data)."""
    report = DATA / "reports" / validate_name(run_name) / "composite_leak.json"
    return [
        module(
            "analysis.composite_leak", "--data", str(DATA / "processed" / "wsts"), "--years", *map(str, years),
            "--workers", str(workers), "--report", str(report),
        )
    ]  # fmt: skip


TRAIN_YEARS, VALIDATION_YEARS = (2018, 2019), (2020,)  # the test year is used only by test_year_stages
TEST_YEARS = (2021,)
# Final configuration of each arm, selected on validation (best.pt = highest validation AP), scored once on the test year.
TEST_ARMS = {
    "supervised": "unet-sup-02",
    "wind": "unet-wind-01",
    "jepa_b": "jepa-02b-probe-r2",
    "jepa_01": "jepa-01-probe",
    "random_encoder": "probe-random-03",
}


def baselines_stages(run_name: str) -> list[list[str]]:
    """Normalisation statistics from the training years, then persistence baselines on validation and training years."""
    reports = DATA / "reports" / validate_name(run_name)
    data = str(DATA / "processed" / "wsts")
    years = lambda ys: [str(y) for y in ys]
    return [
        module("baselines.persistence", "stats", "--data", data, "--years", *years(TRAIN_YEARS),
               "--output", str(reports / "stats-2018-2019.json")),
        module("baselines.persistence", "evaluate", "--data", data, "--years", *years(VALIDATION_YEARS),
               "--report", str(reports / "persistence-2020.json")),
        module("baselines.persistence", "evaluate", "--data", data, "--years", *years(TRAIN_YEARS),
               "--report", str(reports / "persistence-2018-2019.json")),
    ]  # fmt: skip


def train_supervised_stages(run_name: str, *, epochs: int, batch_size: int = 32, workers: int = 4,
                            seed: int = 0, val_every: int = 1, random_frozen_encoder: bool = False) -> list[list[str]]:  # fmt: skip
    """Supervised spread model on train 2018-2019, validated on 2020 every epoch; caches fires on local disk."""
    return [module(
        "train.supervised", "--data", str(DATA / "processed" / "wsts"), "--stats", str(STATS),
        "--cache-dir", "/tmp/fireline-cache", "--output", str(DATA / "runs" / validate_name(run_name)),
        "--epochs", str(epochs), "--batch-size", str(batch_size), "--workers", str(workers),
        "--cache-workers", str(workers), "--seed", str(seed), "--val-every", str(val_every), "--amp", "--device", "cuda",
        *(["--freeze-encoder", "--random-encoder"] if random_frozen_encoder else []),
    )]  # fmt: skip


def train_jepa_stages(run_name: str, *, pretrain_epochs: int = 30, probe_epochs: int = 30, val_every: int = 2,
                      seed: int = 0, weighting: str = "near_fire", target_mode: str = "state") -> list[list[str]]:  # fmt: skip
    """JEPA pretraining on 2018-2019, then the frozen-encoder probe (gate 2c) with the supervised recipe."""
    validate_name(run_name)
    common = ["--data", str(DATA / "processed" / "wsts"), "--stats", str(STATS), "--cache-dir", "/tmp/fireline-cache",
              "--batch-size", "32", "--workers", "4", "--cache-workers", "4", "--seed", str(seed), "--amp", "--device", "cuda"]  # fmt: skip
    pretrain, probe = DATA / "runs" / f"{run_name}-pretrain", DATA / "runs" / f"{run_name}-probe"
    return [
        module("train.jepa", *common, "--output", str(pretrain), "--epochs", str(pretrain_epochs), "--eval-every", "5",
               "--weighting", weighting, "--target-mode", target_mode),
        module("train.supervised", *common, "--output", str(probe), "--epochs", str(probe_epochs),
               "--val-every", str(val_every), "--init-encoder", str(pretrain / "encoder.pt"), "--freeze-encoder"),
    ]  # fmt: skip


def far_fire_stages(run_name: str, *, model_run: str = "unet-sup-02") -> list[list[str]]:
    """Far-fire diagnosis and AP by distance ring (supervised best/last and baselines) on validation 2020."""
    reports = DATA / "reports" / validate_name(run_name)
    data, run = str(DATA / "processed" / "wsts"), DATA / "runs" / validate_name(model_run)
    return [
        module("analysis.far_fire", "--data", data, "--years", *map(str, VALIDATION_YEARS),
               "--report", str(reports / "far-fire-2020.json")),
        module("eval.rings", "--data", data, "--stats", str(STATS), "--cache-dir", "/tmp/fireline-cache",
               "--years", *map(str, VALIDATION_YEARS), "--checkpoint", f"best={run / 'best.pt'}",
               "--checkpoint", f"last={run / 'last.pt'}", "--device", "cuda", "--amp", "--workers", "4",
               "--samples-for", "best", "--report", str(reports / "rings-2020.json")),
    ]  # fmt: skip


def train_probe_stages(run_name: str, *, encoder_run: str, seed: int = 0) -> list[list[str]]:
    """Frozen-encoder probe from an existing pretraining run's encoder.pt, with the train_jepa probe recipe."""
    _, probe = train_jepa_stages("probe-template", seed=seed)
    probe = list(probe)
    probe[probe.index("--output") + 1] = str(DATA / "runs" / validate_name(run_name))
    probe[probe.index("--init-encoder") + 1] = str(DATA / "runs" / validate_name(encoder_run) / "encoder.pt")
    return [probe]


def train_wind_stages(run_name: str, *, baseline_run: str = "unet-sup-02", seed: int = 0) -> list[list[str]]:
    """unet-sup-02 recipe plus wind/distance features, then AP by ring for both models on validation 2020."""
    (train,) = train_supervised_stages(run_name, epochs=30, val_every=2, seed=seed)
    run, base = DATA / "runs" / validate_name(run_name), DATA / "runs" / validate_name(baseline_run)
    rings = module(
        "eval.rings", "--data", str(DATA / "processed" / "wsts"), "--stats", str(STATS),
        "--cache-dir", "/tmp/fireline-cache", "--years", *map(str, VALIDATION_YEARS),
        "--checkpoint", f"wind_best={run / 'best.pt'}", "--checkpoint", f"wind_last={run / 'last.pt'}",
        "--checkpoint", f"sup_best={base / 'best.pt'}", "--checkpoint", f"sup_last={base / 'last.pt'}",
        "--device", "cuda", "--amp", "--workers", "4", "--samples-for", "wind_best",
        "--report", str(DATA / "reports" / run_name / "rings-2020.json"),
    )  # fmt: skip
    return [[*train, "--wind-features"], rings]


V2_VARIANTS: dict[str, list[str]] = {
    "base": [],  # the unet-sup-02 recipe; other seeds measure run-to-run noise
    "far": ["--far-weight", "5", "--far-crop-prob", "0.5", "--far-oversample", "3"],  # far-fire focus
    "history": ["--history-channels"],  # fire-history channels per input day
}


def train_v2_stages(run_name: str, *, variant: str, seed: int = 0, reference_run: str | None = None) -> list[list[str]]:
    """Version 2: the unet-sup-02 recipe with a variant's options, then AP by ring and the coarse far-fire cells
    (8 and 4 px) for the run's best/last checkpoints on validation 2020. ``reference_run``: also evaluate that run's
    best checkpoint in the same job (same hardware and code), e.g. unet-sup-02 as seed 0."""
    if variant not in V2_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(V2_VARIANTS)}")
    (train,) = train_supervised_stages(run_name, epochs=30, val_every=2, seed=seed)
    run = DATA / "runs" / validate_name(run_name)
    checkpoints = ["--checkpoint", f"best={run / 'best.pt'}", "--checkpoint", f"last={run / 'last.pt'}"]
    if reference_run is not None:
        checkpoints += ["--checkpoint", f"reference={DATA / 'runs' / validate_name(reference_run) / 'best.pt'}"]
    rings = module(
        "eval.rings", "--data", str(DATA / "processed" / "wsts"), "--stats", str(STATS),
        "--cache-dir", "/tmp/fireline-cache", "--years", *map(str, VALIDATION_YEARS), *checkpoints,
        "--device", "cuda", "--amp", "--workers", "4", "--samples-for", "best", "--cells", "8", "4",
        "--report", str(DATA / "reports" / run_name / "rings-2020.json"),
    )  # fmt: skip
    return [[*train, *V2_VARIANTS[variant]], rings]


def test_year_stages(run_name: str) -> list[list[str]]:
    """The single test-year evaluation: AP by ring for every arm's best checkpoint and the baselines, plus the far-fire
    diagnosis of 2021. Per-sample counts for the supervised model feed the report's examples."""
    reports = DATA / "reports" / validate_name(run_name)
    data = str(DATA / "processed" / "wsts")
    checkpoints = [
        arg for name, run in TEST_ARMS.items() for arg in ("--checkpoint", f"{name}={DATA / 'runs' / run / 'best.pt'}")
    ]
    return [
        module("eval.rings", "--data", data, "--stats", str(STATS), "--cache-dir", "/tmp/fireline-cache",
               "--years", *map(str, TEST_YEARS), *checkpoints, "--device", "cuda", "--amp", "--workers", "4",
               "--samples-for", "supervised", "--report", str(reports / "rings-2021.json")),
        module("analysis.far_fire", "--data", data, "--years", *map(str, TEST_YEARS),
               "--report", str(reports / "far-fire-2021.json")),
    ]  # fmt: skip


def distance_stages(run_name: str) -> list[list[str]]:
    """Where tomorrow's fire appears relative to today's, on validation and training years (test year not read)."""
    reports = DATA / "reports" / validate_name(run_name)
    data = str(DATA / "processed" / "wsts")
    return [
        module("baselines.persistence", "distance", "--data", data, "--years", *map(str, VALIDATION_YEARS),
               "--report", str(reports / "distance-2020.json")),
        module("baselines.persistence", "distance", "--data", data, "--years", *map(str, TRAIN_YEARS),
               "--report", str(reports / "distance-2018-2019.json")),
    ]  # fmt: skip


# ---------------------------------------------------------------- remote runner


def run_stages(
    run_name: str,
    stages: list[list[str]],
    work_seconds: int,
    *,
    jobs: Path = JOBS,
    commit: Callable[[], None] = lambda: None,
) -> dict:
    """Run stages in order with one shared work budget. Refuses existing jobs; never retries."""
    import os
    import signal
    import subprocess
    import threading
    import time

    validate_name(run_name)
    if not 0 < work_seconds <= MAX_SECONDS:
        raise ValueError(f"work_seconds must be in (0, {MAX_SECONDS}].")
    jobs.mkdir(parents=True, exist_ok=True)
    status_path, log_path = jobs / f"{run_name}.status.json", jobs / f"{run_name}.log"
    if status_path.exists() or log_path.exists():
        raise FileExistsError(f"{run_name} already exists; inspect it rather than overwrite or retry.")
    started = time.monotonic()
    status: dict = {
        "state": "running",
        "stages": [shlex.join(stage) for stage in stages],
        "completed_stages": 0,
        "work_seconds": work_seconds,
    }
    lock = threading.Lock()

    def persist() -> None:
        with lock:
            status["elapsed_seconds"] = round(time.monotonic() - started, 2)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            commit()

    stop_heartbeat = threading.Event()

    def heartbeat() -> None:  # keep the status and volume current during long stages
        while not stop_heartbeat.wait(60):
            persist()

    with status_path.open("x") as stream:
        stream.write(json.dumps(status, indent=2) + "\n")
    process = None
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with log_path.open("x") as log:
            for index, command in enumerate(stages, start=1):
                remaining = work_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Work budget exhausted before the next stage.")
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
                )
                expired = threading.Event()

                def kill(target=process, flag=expired) -> None:
                    flag.set()
                    if target.poll() is None:
                        os.killpg(target.pid, signal.SIGKILL)

                timer = threading.Timer(remaining, kill)
                timer.daemon = True
                timer.start()
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    status["progress"] = line.strip()[:300]
                code = process.wait()
                process.stdout.close()
                timer.cancel()
                if expired.is_set():
                    raise TimeoutError(f"Stage {index} exceeded the work budget and was stopped.")
                if code:
                    raise RuntimeError(f"Stage {index} exited with status {code}; see {log_path.name}. No retry.")
                status["completed_stages"] = index
                persist()
        status["state"] = "complete"
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_heartbeat.set()
        if process is not None and process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
        persist()
    return status


@app.function(
    image=image,
    cpu=(CPU_CORES, CPU_CORES),
    memory=(MEMORY_MIB, MEMORY_MIB),
    timeout=MAX_SECONDS + 300,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
    volumes={str(DATA): data_volume},
)
def run_cpu(run_name: str, stages: list[list[str]], work_seconds: int) -> dict:
    return run_stages(run_name, stages, work_seconds, commit=data_volume.commit)


@app.function(
    image=gpu_image,
    gpu=GPU,
    cpu=(GPU_CPU_CORES, GPU_CPU_CORES),
    memory=(GPU_MEMORY_MIB, GPU_MEMORY_MIB),
    timeout=MAX_SECONDS + 300,
    startup_timeout=900,
    retries=0,
    max_containers=2,  # two independent runs may execute in parallel
    scaledown_window=2,
    single_use_containers=True,
    volumes={str(DATA): data_volume},
)
def run_gpu(run_name: str, stages: list[list[str]], work_seconds: int) -> dict:
    return run_stages(run_name, stages, work_seconds, commit=data_volume.commit)


# ---------------------------------------------------------------- local entrypoints

TIMEOUT_MARGIN = 300


def function_timeout(work_seconds: int) -> int:
    """Modal-enforced hard stop: the runner's work budget plus a fixed margin for startup and the final commit."""
    if not 0 < work_seconds <= MAX_SECONDS:
        raise ValueError(f"work_seconds must be in (0, {MAX_SECONDS}].")
    return work_seconds + TIMEOUT_MARGIN


def current_environment() -> str | None:
    import os

    return os.environ.get("MODAL_ENVIRONMENT") or modal.config.config.get("environment")


def plan_resources(target: modal.Function | None, cpu_cores: int, memory_mib: int) -> dict:
    """Resources a run will get: GPU runs use the decorator's fixed resources; CPU runs the requested ones."""
    if target is run_gpu:
        return {
            "gpu": GPU,
            "cpu_cores": GPU_CPU_CORES,
            "memory_mib": GPU_MEMORY_MIB,
        }
    if not (1 <= cpu_cores <= 8 and 1024 <= memory_mib <= 32768):
        raise ValueError("cpu_cores must be 1..8 and memory_mib 1024..32768.")
    return {"gpu": None, "cpu_cores": cpu_cores, "memory_mib": memory_mib}


def submit(
    run_name: str,
    stages: list[list[str]],
    work_seconds: int,
    confirm_run: bool,
    *,
    cpu_cores: int = CPU_CORES,
    memory_mib: int = MEMORY_MIB,
    target: modal.Function | None = None,
) -> None:
    """Print the plan; with --confirm-run, record and spawn it (the local record prevents accidental relaunch)."""
    validate_name(run_name)
    if current_environment() != ENVIRONMENT:
        raise RuntimeError(f"Run with `modal run -e {ENVIRONMENT} ...` (current: {current_environment()!r}).")
    resources = plan_resources(target, cpu_cores, memory_mib)
    timeout = function_timeout(work_seconds)
    plan = {"run": run_name, "work_seconds": work_seconds, "function_timeout_seconds": timeout, **resources,
            "stages": [shlex.join(s) for s in stages]}  # fmt: skip
    print(json.dumps(plan, indent=2))
    if not confirm_run:
        print("Dry run. Add --confirm-run to start it.")
        return
    record = ROOT / "outputs/modal" / f"{run_name}.json"
    if record.exists():
        raise FileExistsError(f"{record} exists; this run was already submitted.")
    record.parent.mkdir(parents=True, exist_ok=True)
    if target is run_gpu:  # GPU resources stay as declared; only the timeout is set per run
        function = run_gpu.with_options(timeout=timeout)
    else:
        cores, memory = resources["cpu_cores"], resources["memory_mib"]
        function = (target or run_cpu).with_options(timeout=timeout, cpu=(cores, cores), memory=(memory, memory))
    call = function.spawn(run_name, stages, work_seconds)
    record.write_text(json.dumps({**plan, "call_id": call.object_id}, indent=2) + "\n")
    print(f"Submitted {run_name}: {call.object_id} (requires `modal run --detach`; otherwise the run is cancelled)")


@app.local_entrypoint()
def ingest_smoke(run_name: str, year: int = 2019, first: int = 15, workers: int = 8, work_seconds: int = 900,
                 confirm_run: bool = False) -> None:  # fmt: skip
    submit(run_name, ingest_smoke_stages(run_name, year=year, first=first, workers=workers), work_seconds, confirm_run)


@app.local_entrypoint()
def ingest_full(run_name: str, workers: int = 12, cpu_cores: int = 4, memory_mib: int = 12288,
                work_seconds: int = 5400, confirm_run: bool = False) -> None:  # fmt: skip
    stages = ingest_full_stages(run_name, workers=workers)
    submit(run_name, stages, work_seconds, confirm_run, cpu_cores=cpu_cores, memory_mib=memory_mib)


@app.local_entrypoint()
def composite_leak(run_name: str, workers: int = 2, cpu_cores: int = 2, memory_mib: int = 4096,
                   work_seconds: int = 1200, confirm_run: bool = False) -> None:  # fmt: skip
    stages = composite_leak_stages(run_name, workers=workers)
    submit(run_name, stages, work_seconds, confirm_run, cpu_cores=cpu_cores, memory_mib=memory_mib)


@app.local_entrypoint()
def baselines(run_name: str, work_seconds: int = 1200, confirm_run: bool = False) -> None:
    submit(run_name, baselines_stages(run_name), work_seconds, confirm_run)


@app.local_entrypoint()
def train_supervised(run_name: str, epochs: int, batch_size: int = 32, seed: int = 0, val_every: int = 1,
                     random_frozen_encoder: bool = False, work_seconds: int = 1800, confirm_run: bool = False) -> None:  # fmt: skip
    stages = train_supervised_stages(run_name, epochs=epochs, batch_size=batch_size, seed=seed, val_every=val_every,
                                     random_frozen_encoder=random_frozen_encoder)  # fmt: skip
    submit(run_name, stages, work_seconds, confirm_run, target=run_gpu)


@app.local_entrypoint()
def distance(run_name: str, work_seconds: int = 900, confirm_run: bool = False) -> None:
    submit(run_name, distance_stages(run_name), work_seconds, confirm_run)


@app.local_entrypoint()
def train_jepa(run_name: str, pretrain_epochs: int = 30, probe_epochs: int = 30, val_every: int = 2, seed: int = 0,
               weighting: str = "near_fire", target_mode: str = "state", work_seconds: int = 7200,
               confirm_run: bool = False) -> None:  # fmt: skip
    stages = train_jepa_stages(run_name, pretrain_epochs=pretrain_epochs, probe_epochs=probe_epochs,
                               val_every=val_every, seed=seed, weighting=weighting, target_mode=target_mode)  # fmt: skip
    submit(run_name, stages, work_seconds, confirm_run, target=run_gpu)


@app.local_entrypoint()
def far_fire(run_name: str, work_seconds: int = 1800, confirm_run: bool = False) -> None:
    submit(run_name, far_fire_stages(run_name), work_seconds, confirm_run, target=run_gpu)


@app.local_entrypoint()
def train_wind(run_name: str, seed: int = 0, work_seconds: int = 4500, confirm_run: bool = False) -> None:
    submit(run_name, train_wind_stages(run_name, seed=seed), work_seconds, confirm_run, target=run_gpu)


@app.local_entrypoint()
def train_probe(run_name: str, encoder_run: str, seed: int = 0, work_seconds: int = 3600,
                confirm_run: bool = False) -> None:  # fmt: skip
    submit(run_name, train_probe_stages(run_name, encoder_run=encoder_run, seed=seed), work_seconds, confirm_run,
           target=run_gpu)  # fmt: skip


@app.local_entrypoint()
def train_v2(run_name: str, variant: str, seed: int = 0, reference_run: str = "", work_seconds: int = 4500,
             confirm_run: bool = False) -> None:  # fmt: skip
    stages = train_v2_stages(run_name, variant=variant, seed=seed, reference_run=reference_run or None)
    submit(run_name, stages, work_seconds, confirm_run, target=run_gpu)


@app.local_entrypoint()
def test_year(run_name: str, work_seconds: int = 2400, confirm_run: bool = False) -> None:
    submit(run_name, test_year_stages(run_name), work_seconds, confirm_run, target=run_gpu)
