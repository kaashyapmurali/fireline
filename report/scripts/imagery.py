"""Report imagery: the interactive timeline layers, the GIF and the filmstrip, from one fire chosen by a fixed rule.

Selection rule (no hand-picking): among fires of the evaluated year, take the fire with the most target days that have
at least ``MIN_DAY_FIRE`` fire pixels tomorrow; ties go to the larger total fire. All of that fire's target days are
rendered. The supervised model's predictions are computed here from its checkpoint (same code as evaluation).

Per day, transparent PNG layers share one crop (the fire's extent over all rendered days, plus a margin):
``base.png`` (terrain hillshade, once), ``day-<t>-today.png`` (today's fire), ``day-<t>-truth.png`` (tomorrow's fire,
outline), ``day-<t>-prediction.png`` (probability heat), ``day-<t>-outcome.png`` (hits / misses / false alarms at 0.5).
Every layer is upscaled by an integer factor with nearest-neighbour sampling, so pixels stay square and true to 375 m.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import views
from PIL import Image, ImageDraw, ImageFont
from samples import MODEL_DAY_RULE, select_model_day

from fireline.data.cache import materialize
from fireline.data.convert import load_fire
from fireline.data.inputs import Stats
from fireline.eval.fire_batches import build_fire, predict_fire
from fireline.models.spread import SpreadModel

MIN_DAY_FIRE = 50
BIG_DAY = 500
MARGIN = 12
SCALE = 4
GIF_WIDTH = 480
COLORS = {  # RGBA, tuned for both website themes
    "today": (234, 120, 30, 255),
    "truth": (196, 30, 40, 255),
    "hit": (46, 160, 67, 230),
    "miss": (214, 39, 40, 230),
    "false_alarm": (31, 119, 180, 230),
}


def select_fire(per_sample: list[dict]) -> str:
    days, total = defaultdict(int), defaultdict(int)
    for row in per_sample:
        fire_tomorrow = row["near_pos"] + row["far_pos"]
        total[row["fire"]] += fire_tomorrow
        days[row["fire"]] += fire_tomorrow >= MIN_DAY_FIRE
    return max(days, key=lambda f: (days[f], total[f]))


def hillshade(elevation: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Grey hillshade in [0, 1] from an elevation grid (375 m pixels)."""
    elevation = np.nan_to_num(elevation, nan=float(np.nanmean(elevation)))
    dy, dx = np.gradient(elevation, 375.0)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shade + 1) / 2, 0, 1)


def _rgba(mask: np.ndarray, color: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[mask] = color
    return out


def _outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~interior


def _heat(prob: np.ndarray) -> np.ndarray:
    """Probability as a blue haze (distinct from the fire colours); opacity rises with probability, none below 0.05."""
    p = np.clip(prob, 0, 1)
    out = np.zeros((*p.shape, 4), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = 31, 119, 180
    out[..., 3] = (255 * np.where(p < 0.05, 0, 0.15 + 0.7 * p)).astype(np.uint8)
    return out


def _save(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(array)
    image = image.resize((image.width * SCALE, image.height * SCALE), Image.Resampling.NEAREST)
    image.save(path, optimize=True)


def render_timeline(
    npz: Path, checkpoint: Path, stats: Stats, out_dir: Path, cache: Path, device: torch.device
) -> tuple[dict, dict[int, np.ndarray], np.ndarray]:
    """Timeline layers; returns the summary, the probability map per target day, and the cropped terrain shading."""
    record = load_fire(npz)
    state = torch.load(checkpoint, map_location="cpu")
    model = SpreadModel(history=5)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    fire = build_fire(materialize(npz, cache), stats, history=5)
    height, width = record.fire_hour.shape[1:]
    mask = record.fire_mask
    probs = {t: p[:height, :width] for ts, ps in predict_fire(model, fire, device, batch_size=2)
             for t, p in zip(ts, ps.cpu().numpy(), strict=True)}  # fmt: skip
    targets = sorted(probs)
    union = np.zeros((height, width), dtype=bool)
    for t in targets:
        union |= mask[t] | mask[t + 1] | (probs[t] >= 0.5)
    ys, xs = np.nonzero(union)
    y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN + 1, height)
    x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN + 1, width)
    crop = (slice(y0, y1), slice(x0, x1))

    out_dir.mkdir(parents=True, exist_ok=True)
    shade = hillshade(record.channel("elevation"))[crop]
    low, high = np.percentile(shade, [2, 98])
    shade = np.clip((shade - low) / max(high - low, 1e-6), 0, 1)  # stretch: terrain readable, still light
    base = np.stack([(150 + 95 * shade)] * 3 + [np.full(shade.shape, 255.0)], axis=-1).astype(np.uint8)
    _save(base, out_dir / "timeline-base.png")
    days = []
    for t in targets:
        today, truth, prob = mask[t][crop], mask[t + 1][crop], probs[t][crop]
        flagged = prob >= 0.5
        outcome = np.zeros((*today.shape, 4), dtype=np.uint8)
        outcome[flagged & truth] = COLORS["hit"]
        outcome[~flagged & truth] = COLORS["miss"]
        outcome[flagged & ~truth] = COLORS["false_alarm"]
        _save(_rgba(today, COLORS["today"]), out_dir / f"timeline-{t}-today.png")
        _save(_rgba(_outline(truth), COLORS["truth"]), out_dir / f"timeline-{t}-truth.png")
        _save(_heat(prob), out_dir / f"timeline-{t}-prediction.png")
        _save(outcome, out_dir / f"timeline-{t}-outcome.png")
        days.append({"t": t, "date": record.dates[t], "next_date": record.dates[t + 1],
                     "fire_today": int(today.sum()), "fire_tomorrow": int(truth.sum()),
                     "hits": int((flagged & truth).sum()), "misses": int((~flagged & truth).sum()),
                     "false_alarms": int((flagged & ~truth).sum())})  # fmt: skip
    summary = {"fire": record.fire_id, "year": record.year, "crop_px": [int(y1 - y0), int(x1 - x0)], "scale": SCALE,
            "crop_box": [int(y0), int(y1), int(x0), int(x1)],
            "pixel_m": 375, "days": days}  # fmt: skip
    return summary, probs, shade


def compose_frames(timeline: dict, layer_dir: Path) -> list[Image.Image]:
    """Base + today's fire (light) + tomorrow's outcome (hit / miss / false alarm), with a caption, one frame per day.

    Static frames use the outcome encoding because three overlapping layers do not read well without toggles; the
    website's interactive timeline shows the separate layers."""
    frames = []
    base = Image.open(layer_dir / "timeline-base.png").convert("RGBA")
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for day in timeline["days"]:
        frame = base.copy()
        today = Image.open(layer_dir / f"timeline-{day['t']}-today.png").convert("RGBA")
        today.putalpha(today.getchannel("A").point(lambda a: a * 110 // 255))
        frame.alpha_composite(today)
        frame.alpha_composite(Image.open(layer_dir / f"timeline-{day['t']}-outcome.png").convert("RGBA"))
        canvas = Image.new("RGBA", (frame.width, frame.height + 34), (248, 246, 240, 255))
        canvas.alpha_composite(frame, (0, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"{day['date']} → {day['next_date']}  orange today · green hit · red missed · blue false alarm",
                  fill=(26, 27, 26, 255), font=font)  # fmt: skip
        frames.append(canvas.convert("RGB"))
    return frames


FILMSTRIP_RULE = (
    f"start: first day with at least {MIN_DAY_FIRE} fire pixels tomorrow; largest day: most fire tomorrow; among days with "
    f"at least {BIG_DAY} pixels tomorrow, best: highest F1, typical: median F1; shown in date order"
)


def _f1(day: dict) -> float:
    total = 2 * day["hits"] + day["false_alarms"] + day["misses"]
    return 2 * day["hits"] / total if total else 0.0


def filmstrip_picks(days: list[dict]) -> dict[int, str]:
    """Index -> role for the story panels; a later role never reuses a day an earlier role took."""
    roles: dict[int, str] = {}
    roles[next(i for i, d in enumerate(days) if d["fire_tomorrow"] >= MIN_DAY_FIRE)] = "start"
    roles.setdefault(max(range(len(days)), key=lambda i: days[i]["fire_tomorrow"]), "blow-up")
    big = sorted((i for i, d in enumerate(days) if d["fire_tomorrow"] >= BIG_DAY and i not in roles),
                 key=lambda i: _f1(days[i]))  # fmt: skip
    if big:
        roles[big[-1]] = "best"
        rest = big[:-1]
        if rest:
            roles[rest[len(rest) // 2]] = "typical"
    return roles


def write_gif_and_filmstrip(timeline: dict, layer_dir: Path, out_dir: Path) -> dict:
    """The GIF shows every active day (fire today or tomorrow); the filmstrip shows the story panels of ``FILMSTRIP_RULE``."""
    active = [d for d in timeline["days"] if d["fire_today"] + d["fire_tomorrow"] > 0]
    frames = compose_frames({**timeline, "days": active}, layer_dir)
    gif = out_dir / "timeline.gif"
    factor = min(1.0, GIF_WIDTH / frames[0].width)  # README / social size; the website uses the full-size layers
    small = [f.resize((round(f.width * factor), round(f.height * factor)), Image.Resampling.LANCZOS) for f in frames]
    small = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in small]
    small[0].save(gif, save_all=True, append_images=small[1:], duration=900, loop=0, optimize=True)
    roles = filmstrip_picks(active)
    picks = sorted(roles)
    strip = Image.new("RGB", (sum(frames[i].width for i in picks) + 8 * (len(picks) - 1), frames[0].height),
                      (248, 246, 240))  # fmt: skip
    x = 0
    for i in picks:
        strip.paste(frames[i], (x, 0))
        x += frames[i].width + 8
    strip_path = out_dir / "filmstrip.png"
    strip.save(strip_path, optimize=True)
    return {
        "gif": gif.name,
        "gif_days": [d["t"] for d in active],
        "filmstrip": strip_path.name,
        "filmstrip_days": [active[i]["t"] for i in picks],
        "filmstrip_roles": [roles[i] for i in picks],
        "filmstrip_rule": FILMSTRIP_RULE,
    }


def build(per_sample_report: Path, data_dir: Path, year: int, checkpoint: Path, stats: Stats, bundle: Path,
          cache: Path) -> dict:  # fmt: skip
    rows = json.loads(per_sample_report.read_text())["per_sample"]
    fire_id = select_fire(rows)
    npz = data_dir / str(year) / f"{fire_id}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} missing: download it from the volume (processed/wsts/{year}/{fire_id}.npz).")
    device = torch.device("cpu")
    layers = bundle / "images"
    timeline, probs, shade = render_timeline(npz, checkpoint, stats, layers, cache, device)
    timeline.update(write_gif_and_filmstrip(timeline, layers, bundle / "images"))
    typical = timeline["filmstrip_days"][timeline["filmstrip_roles"].index("typical")]
    model_day = select_model_day(load_fire(npz), sorted(probs), typical, timeline["crop_box"])
    timeline["model_view"] = views.render_model(npz, checkpoint, stats, cache, model_day, timeline["crop_box"], layers,
                                                shade)  # fmt: skip
    timeline["model_view"]["selection_rule"] = MODEL_DAY_RULE
    timeline["model_view"]["reference_typical_t"] = typical
    timeline["explainer"] = views.render_explainer(npz, probs[typical], typical, timeline["crop_box"], layers)
    timeline["selection_rule"] = (f"fire with the most target days having at least {MIN_DAY_FIRE} fire pixels "
                                  "tomorrow; ties to the larger total fire")  # fmt: skip
    return timeline
