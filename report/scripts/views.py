"""Report views beyond the timeline, all from the timeline fire (no hand-picked imagery):

- ``model``: what the network sees and computes on the timeline's "typical" day: six input channels, the encoder's
  feature maps (full resolution and 1/8) for each of the five input days, and the fused 1/8 features. Feature maps
  are shown as RGB of their first three principal components (one PCA basis per scale, shared across days, so colours
  are comparable between days).
- ``explainer``: one RGBA PNG at native resolution for the same day (R = predicted probability, G = tomorrow's fire,
  B = today's fire) that the website's threshold explorer reads to compute precision and recall in the browser.
- ``leak``: the composite update (day ``u``) of this fire with the most pixels that burn in the following 15 days:
  the NDVI change at ``u`` and when each pixel burns relative to ``u``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from fireline.data.cache import materialize
from fireline.data.convert import load_fire
from fireline.data.inputs import DYNAMIC_FEATURES, Stats, raw_dynamic
from fireline.eval.fire_batches import build_fire
from fireline.models.spread import SpreadModel, encode_days

THUMB_SCALE = 2
WINDOW = 15
NDVI_SCALE = 1e4  # VNP13A1 values are stored x 1e4
INPUTS = (  # (key, label, colormap, unit)
    ("active_fire", "Active fire", None, ""),
    ("ndvi", "Vegetation (NDVI)", "YlGn", ""),
    ("energy_release_component", "Energy release component", "YlOrRd", ""),
    ("temperature_max", "Max temperature", "inferno", "K"),
    ("wind_speed", "Wind speed", "PuBu", "m/s"),
    ("elevation", "Elevation (shaded)", None, "m"),
)


def _save(rgb: np.ndarray, path: Path, scale: int = THUMB_SCALE) -> None:
    image = Image.fromarray(rgb)
    image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST).save(path, optimize=True)


def _colour(values: np.ndarray, cmap: str, valid: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(values[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    rgb = (colormaps[cmap](np.nan_to_num(scaled))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 235
    return rgb


def _pca_rgb(maps: list[np.ndarray]) -> list[np.ndarray]:
    """RGB images from the first three principal components of feature maps (C, h, w); one basis for all maps."""
    flat = np.concatenate([m.reshape(m.shape[0], -1).T for m in maps])
    mean = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    projected = (flat - mean) @ vt[:3].T
    low, high = np.percentile(projected, [2, 98], axis=0)
    projected = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
    out, start = [], 0
    for m in maps:
        n = m.shape[1] * m.shape[2]
        out.append((projected[start : start + n].reshape(m.shape[1], m.shape[2], 3) * 255).astype(np.uint8))
        start += n
    return out


def render_model(npz: Path, checkpoint: Path, stats: Stats, cache: Path, t: int, crop_box: list[int], out: Path,
                 shade: np.ndarray) -> dict:  # fmt: skip
    """Input thumbnails and feature maps for target day ``t`` (inputs: days t-4..t)."""
    record = load_fire(npz)
    y0, y1, x0, x1 = crop_box
    crop = (slice(y0, y1), slice(x0, x1))
    raw = raw_dynamic(record, [t])[0]
    valid = np.ones((y1 - y0, x1 - x0), dtype=bool)
    files = []
    for key, label, cmap, _unit in INPUTS:
        name = f"model-input-{key.replace('_', '-')}.png"
        if key == "active_fire":
            rgb = np.full((*valid.shape, 3), 240, dtype=np.uint8)
            rgb[record.fire_mask[t][crop]] = (234, 120, 30)
        elif key == "elevation":
            rgb = np.stack([(150 + 95 * shade)] * 3, axis=-1).astype(np.uint8)
        elif key == "wind_speed":
            u, v = (raw[DYNAMIC_FEATURES.index(k)][crop] for k in ("wind_u", "wind_v"))
            rgb = _colour(np.hypot(u, v), cmap, np.isfinite(u))
        else:
            values = raw[DYNAMIC_FEATURES.index(key)][crop]
            rgb = _colour(values, cmap, np.isfinite(values))
        _save(rgb, out / name)
        missing = key not in ("active_fire", "elevation", "wind_speed") and bool(
            np.isnan(raw[DYNAMIC_FEATURES.index(key)][crop]).all()
        )
        files.append({"key": key, "label": label, "file": name, "missing": missing})

    model = SpreadModel(history=5)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.eval()
    fire = build_fire(materialize(npz, cache), stats, history=5)
    with torch.no_grad():
        dynamic = fire.dynamic[t - 4 : t + 1][None].float()
        encoded = encode_days(model.encoder, dynamic, fire.static[None].float())
        fused = model.fuse[-1](encoded[-1].flatten(1, 2))[0].numpy()
    full = [encoded[0][0, d].numpy()[:, y0:y1, x0:x1] for d in range(5)]
    s8 = (slice(y0 // 8, -(-y1 // 8)), slice(x0 // 8, -(-x1 // 8)))
    coarse = [encoded[-1][0, d].numpy()[:, s8[0], s8[1]] for d in range(5)]
    feature_files = {"full": [], "coarse": []}
    for kind, maps, scale in (("full", full, THUMB_SCALE), ("coarse", coarse, 8 * THUMB_SCALE)):
        for d, rgb in enumerate(_pca_rgb(maps)):
            name = f"model-features-{kind}-day{d}.png"
            _save(rgb, out / name, scale)
            feature_files[kind].append(name)
    fused_rgb = _pca_rgb([fused[:, s8[0], s8[1]]])[0]
    _save(fused_rgb, out / "model-fused.png", 8 * THUMB_SCALE)
    return {"t": t, "date": record.dates[t], "next_date": record.dates[t + 1], "inputs": files,
            "features_full": feature_files["full"], "features_coarse": feature_files["coarse"],
            "fused": "model-fused.png", "input_dates": [record.dates[d] for d in range(t - 4, t + 1)]}  # fmt: skip


def render_explainer(npz: Path, probs: np.ndarray, t: int, crop_box: list[int], out: Path) -> dict:
    """R = probability, G = tomorrow's fire, B = today's fire, at native resolution (one pixel = 375 m)."""
    record = load_fire(npz)
    y0, y1, x0, x1 = crop_box
    crop = (slice(y0, y1), slice(x0, x1))
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[..., 0] = np.round(np.clip(probs[crop], 0, 1) * 255).astype(np.uint8)
    rgba[..., 1] = record.fire_mask[t + 1][crop] * 255
    rgba[..., 2] = record.fire_mask[t][crop] * 255
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(out / "explainer-day.png", optimize=True)
    return {"t": t, "date": record.dates[t], "next_date": record.dates[t + 1], "file": "explainer-day.png",
            "size": [int(y1 - y0), int(x1 - x0)]}  # fmt: skip


def render_leak(npz: Path, crop_box: list[int], out: Path, shade: np.ndarray) -> dict:
    """NDVI change at the composite update with the most in-window burns, and burn timing relative to it."""
    record = load_fire(npz)
    y0, y1, x0, x1 = crop_box
    crop = (slice(y0, y1), slice(x0, x1))
    k = DYNAMIC_FEATURES.index("ndvi")  # one day at a time: the full stack would not fit in memory comfortably
    ndvi = np.stack([raw_dynamic(record, [d])[0, k][crop] for d in range(len(record.dates))]) / NDVI_SCALE
    mask = record.fire_mask[:, y0:y1, x0:x1]
    burn_day = np.where(mask.any(0), mask.argmax(0), -1)
    diff = np.abs(np.nan_to_num(ndvi[1:] - ndvi[:-1]))
    # a composite update changes most pixels at once (every 8 days); partial changes are not counted as updates
    updates = [u for u in range(1, len(ndvi)) if (diff[u - 1] > 1e-6).mean() > 0.5]
    count = {u: int(((burn_day > u) & (burn_day <= u + WINDOW)).sum()) for u in updates}
    u = max(count, key=lambda k: count[k])
    change = ndvi[u] - ndvi[u - 1]
    valid = np.isfinite(change)
    limit = float(np.nanpercentile(np.abs(change[valid]), 98))
    rgb = (colormaps["BrBG"](np.clip(change / max(limit, 1e-6) / 2 + 0.5, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 235
    _save(rgb, out / "leak-ndvi-change.png")
    timing = np.stack([(150 + 95 * shade)] * 3, axis=-1).astype(np.uint8)
    groups = {"already_burned": (burn_day >= 0) & (burn_day <= u),
              "burns_in_window": (burn_day > u) & (burn_day <= u + WINDOW),
              "burns_later": burn_day > u + WINDOW}  # fmt: skip
    colours = {"already_burned": (90, 90, 80), "burns_in_window": (214, 39, 40), "burns_later": (240, 170, 60)}
    for name, where in groups.items():
        timing[where] = colours[name]
    _save(timing, out / "leak-burn-timing.png")
    never = (burn_day < 0) & valid
    medians = {name: float(np.nanmedian(change[where & valid])) if (where & valid).any() else None
               for name, where in {**groups, "never_burns": never}.items()}  # fmt: skip
    return {"update_date": record.dates[u], "previous_date": record.dates[u - 1],
            "window_end": record.dates[min(u + WINDOW, len(record.dates) - 1)],
            "pixels": {k: int(v.sum()) for k, v in {**groups, "never_burns": never}.items()},
            "median_change": medians, "colour_limit": limit,
            "files": {"change": "leak-ndvi-change.png", "timing": "leak-burn-timing.png"}}  # fmt: skip
