"""Report views beyond the timeline, all from the timeline fire (no hand-picked imagery):

- ``model``: what the network sees and computes on the nearest input-complete day to the timeline's "typical" day
  (fixed coverage rule in samples.py): all daily/static inputs, the encoder's
  whole-block encoder outputs at all four scales (full and 1/8 for all five days, 1/2 and 1/4 for today), fused
  maps at every scale, and whole decoder-block outputs. Feature maps are RGB of their first three principal components
  (one encoder PCA basis per scale, shared across days). Architecture shapes and parameter counts are model-derived.
- ``explainer``: one RGBA PNG at native resolution for the unchanged typical day (R = predicted probability, G = tomorrow's fire,
  B = today's fire) that the website's threshold explorer reads to compute precision and recall in the browser.
- ``leak``: the composite update (day ``u``) of this fire with the most pixels that burn in the following 15 days:
  the NDVI change at ``u`` and when each pixel burns relative to ``u``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from architecture import architecture_facts, capture_decode, last_fire_channel
from gallery import GALLERY, landcover_rgb, physical_values, value_summary
from matplotlib import colormaps
from PIL import Image, ImageDraw
from wind import wind_arrows

from fireline.data.cache import materialize
from fireline.data.convert import FireRecord, load_fire
from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES, Stats, raw_dynamic, raw_static
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
    ("wind_speed", "Wind speed + direction", "PuBu", "m/s"),
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


def _wind_overlay(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Daily-mean flow arrows over the speed colours, with a white halo for contrast."""
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in wind_arrows(u, v):
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        dx, dy = dx / length, dy / length
        draw.line((x0, y0, x1, y1), fill="white", width=3)
        draw.polygon([(x1 + dx, y1 + dy), (x1 - 7 * dx - 4 * dy, y1 - 7 * dy + 4 * dx),
                      (x1 - 7 * dx + 4 * dy, y1 - 7 * dy - 4 * dx)], fill="white")
        draw.line((x0, y0, x1, y1), fill=(20, 35, 45), width=1)
        draw.polygon([(x1, y1), (x1 - 5 * dx - 2.5 * dy, y1 - 5 * dy + 2.5 * dx),
                      (x1 - 5 * dx + 2.5 * dy, y1 - 5 * dy - 2.5 * dx)], fill=(20, 35, 45))
    return np.asarray(image)


def render_gallery(record: FireRecord, raw: np.ndarray, crop_box: list[int], out: Path) -> list[dict]:
    """All physical inputs for the model-view day/crop; extra diagnostic views are explicitly named."""
    y0, y1, x0, x1 = crop_box
    crop = (slice(y0, y1), slice(x0, x1))
    static = raw_static(record, window=crop)

    def values(key: str) -> np.ndarray:
        if key in DYNAMIC_FEATURES:
            array = raw[DYNAMIC_FEATURES.index(key)][crop]
        elif key in STATIC_FEATURES:
            array = static[STATIC_FEATURES.index(key)]
        elif key == "wind_speed":
            array = np.hypot(raw[DYNAMIC_FEATURES.index("wind_u")][crop], raw[DYNAMIC_FEATURES.index("wind_v")][crop])
        else:  # raw aspect and categorical land cover (represented by derived channels in the model)
            array = record.channel(key)[crop]
        return physical_values(key, array)

    groups = []
    for group, definitions in GALLERY:
        tiles = []
        for key, label, source, resolution, unit, cmap in definitions:
            legend, ranges = [], []
            if key == "false_colour":
                bands = [values(k) for k in ("viirs_m11", "viirs_i2", "viirs_i1")]
                valid = np.isfinite(np.stack(bands)).all(axis=0)
                ranges = [{"band": name, **value_summary(b)} for name, b in zip(("R: M11", "G: I2", "B: I1"), bands, strict=True)]
                summary = value_summary(np.stack(bands))
                summary.update(missing=not bool(valid.any()), coverage=float(valid.mean()),
                               uniform=bool(valid.any()) and all(r["uniform"] for r in ranges))
                rgb = np.stack([_colour(b, "gray", valid)[..., 0] for b in bands], axis=-1)
                rgb[~valid] = 235
            else:
                array = values(key)
                if key == "landcover":
                    array = np.where(np.isin(array, np.arange(1, 18)), array, np.nan)
                summary = value_summary(array)
                if key == "landcover":
                    rgb, legend = landcover_rgb(array)
                elif summary["uniform"]:
                    colour = (np.array(colormaps[cmap](0.5)[:3]) * 255).astype(np.uint8)
                    rgb = np.broadcast_to(colour, (*array.shape, 3)).copy()
                    rgb[~np.isfinite(array)] = 235
                else:
                    rgb = _colour(array, cmap, np.isfinite(array))
                if key == "wind_speed":
                    rgb = _wind_overlay(rgb, raw[DYNAMIC_FEATURES.index("wind_u")][crop], raw[DYNAMIC_FEATURES.index("wind_v")][crop])
            name = f"gallery-{key.replace('_', '-')}.png"
            _save(rgb, out / name)
            tiles.append({"key": key, "name": label, "source": source, "resolution": resolution, "unit": unit,
                          "file": name, **summary, "legend": legend, "ranges": ranges})
        groups.append({"name": group, "tiles": tiles})
    return groups


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
    gallery = render_gallery(record, raw, crop_box, out)
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
            rgb = _colour(np.hypot(u, v), cmap, np.isfinite(u) & np.isfinite(v))
            rgb = _wind_overlay(rgb, u, v)
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
        _, fused_maps, decoded_maps = capture_decode(model, encoded, last_fire_channel(dynamic))
        fused = fused_maps[3][0].numpy()
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
    architecture = architecture_facts(model)
    for i, stage in enumerate(architecture["scales"]):
        factor, kind = stage["factor"], stage["name"]
        region = (slice(y0 // factor, -(-y1 // factor)), slice(x0 // factor, -(-x1 // factor)))
        if i in (0, 3):  # preserve the current full/coarse images and their shared five-day PCA basis byte-for-byte
            stage["features_today"] = feature_files["full" if i == 0 else "coarse"][-1]
        else:
            maps = [encoded[i][0, d].numpy()[:, region[0], region[1]] for d in range(5)]
            name = f"model-features-{kind}-today.png"
            _save(_pca_rgb(maps)[-1], out / name, factor * THUMB_SCALE)
            stage["features_today"] = name
        if i == 3:
            stage["fused_file"] = "model-fused.png"
        else:
            name = f"model-fused-{kind}.png"
            _save(_pca_rgb([fused_maps[i][0].numpy()[:, region[0], region[1]]])[0], out / name, factor * THUMB_SCALE)
            stage["fused_file"] = name
    for step in architecture["decoder"]:
        i = step["index"]
        factor = 2 ** i
        region = (slice(y0 // factor, -(-y1 // factor)), slice(x0 // factor, -(-x1 // factor)))
        name = f"model-decoder-{step['name']}.png"
        _save(_pca_rgb([decoded_maps[i][0].numpy()[:, region[0], region[1]]])[0], out / name, factor * THUMB_SCALE)
        step["file"] = name
    return {"t": t, "date": record.dates[t], "next_date": record.dates[t + 1], "inputs": files,
            "features_full": feature_files["full"], "features_coarse": feature_files["coarse"],
            "fused": "model-fused.png", "architecture": architecture, "gallery": gallery, "input_dates": [record.dates[d] for d in range(t - 4, t + 1)]}  # fmt: skip


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
