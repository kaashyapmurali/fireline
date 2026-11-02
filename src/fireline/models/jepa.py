"""JEPA for wildfire spread: predict tomorrow's latent grid from the previous days' latent grids and the forecast.

- Context encoder: the same per-day ``DayEncoder`` as the supervised ``SpreadModel`` (stride-8 latent = last stage).
- Target encoder: an exponential-moving-average copy of the context encoder, never trained by gradients.
- Predictor: convolutional, at stride 8. Input: the ``history`` context latents stacked over days plus tomorrow's
  GFS forecast channels (issued before tomorrow's detections; checked for leakage), average-pooled
  to stride 8. Output: the predicted latent of day t + 1.
- Loss: per-cell squared error against the layer-normalised target latent, weighted towards cells near fire (today or
  tomorrow), plus VICReg-style variance and covariance terms on the context latents as a guard against collapse.
After pretraining, the context encoder is frozen inside ``SpreadModel`` and only fusion + decoder are trained.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from fireline.data.inputs import DYNAMIC_FEATURES
from fireline.models.spread import FIRE_CHANNEL, ConvBlock, DayEncoder, encode_days

FORECAST_CHANNELS = tuple(
    DYNAMIC_FEATURES.index(n)
    for n in ("forecast_precipitation", "forecast_wind_speed", "forecast_temperature", "forecast_specific_humidity")
)
STRIDE = 8


def channel_layer_norm(x: torch.Tensor) -> torch.Tensor:
    """Layer norm over the channel dimension of (B, C, h, w) maps, without learned parameters."""
    return F.layer_norm(x.movedim(1, -1), x.shape[1:2]).movedim(-1, 1)


class Predictor(nn.Module):
    """Residual predictor: normalised last-day context latent + a learned change (zero at initialisation).

    Starting exactly at "tomorrow = today" makes the copy baseline the starting point, so gate 2b measures whether the
    predictor learned anything beyond copying.
    """

    def __init__(self, history: int, latent: int = 128, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(history * latent + len(FORECAST_CHANNELS), width, 1), nn.GELU())
        self.blocks = nn.Sequential(*[ConvBlock(width, width) for _ in range(depth)])
        self.out = nn.Conv2d(width, latent, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def change(self, context: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        """The learned change (B, C, h, w); zero at initialisation."""
        return self.out(self.blocks(self.inp(torch.cat([context.flatten(1, 2), forecast], dim=1))))

    def forward(self, context: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        """``context`` (B, T, C, h, w) latents; ``forecast`` (B, F, h, w) pooled tomorrow's forecast -> (B, C, h, w)."""
        return channel_layer_norm(context[:, -1]) + self.change(context, forecast)


def near_fire_cells(fire_today: torch.Tensor, fire_tomorrow: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """(B, H, W) masks -> (B, h, w) bool: cells within ``radius`` cells of fire on either day (at stride 8)."""
    fire = torch.maximum(fire_today, fire_tomorrow).float()[:, None]
    cells = F.max_pool2d(fire, STRIDE, STRIDE)
    return F.max_pool2d(cells, 2 * radius + 1, 1, radius)[:, 0] > 0


def change_cells(fire_today: torch.Tensor, fire_tomorrow: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """(B, H, W) masks -> (B, h, w) bool: cells within ``radius`` cells of a pixel whose fire state changes
    between the two days (fire appears or disappears); persistent fire and no-fire pixels do not count."""
    changed = (fire_today != fire_tomorrow).float()[:, None]
    cells = F.max_pool2d(changed, STRIDE, STRIDE)
    return F.max_pool2d(cells, 2 * radius + 1, 1, radius)[:, 0] > 0


WEIGHTINGS = ("near_fire", "change")
TARGET_MODES = ("state", "change")


def variance_covariance(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg terms on latents (N, C): hinge on per-channel std below 1, and mean squared off-diagonal covariance."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + eps)
    variance = F.relu(1 - std).mean()
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off_diagonal = cov - torch.diag(torch.diag(cov))
    return variance, (off_diagonal**2).sum() / z.shape[1]


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy) of the normalised singular values of centred latents (N, C); between 1 and C."""
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))


class SpreadJEPA(nn.Module):
    def __init__(self, history: int = 5, widths: tuple[int, int, int, int] = (32, 64, 128, 128),
                 near_fire_weight: float = 10.0, variance_weight: float = 1.0, covariance_weight: float = 0.04,
                 weighting: str = "near_fire", target_mode: str = "state") -> None:  # fmt: skip
        """``weighting``: "near_fire" (jepa-01: cells near fire today or tomorrow) or "change" (cells where fire
        appears or disappears). ``target_mode``: "state" (jepa-01: predict tomorrow's latent; residual from the
        online encoder's latent of today; absolute loss) or "change" (predict tomorrow's latent as the target
        encoder's latent of today plus a learned change; loss relative to the copy-today error, so making days
        look alike does not lower it)."""
        super().__init__()
        if weighting not in WEIGHTINGS or target_mode not in TARGET_MODES:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, target_mode one of {TARGET_MODES}.")
        self.weighting, self.target_mode = weighting, target_mode
        self.history = history
        self.encoder = DayEncoder(widths=widths)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = Predictor(history, latent=widths[-1])
        self.near_fire_weight, self.variance_weight, self.covariance_weight = (
            near_fire_weight, variance_weight, covariance_weight)  # fmt: skip

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(momentum).add_(online.detach(), alpha=1 - momentum)

    def latents(self, dynamic: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        """``dynamic`` holds history + 1 days (the last is day t + 1). Returns context, prediction, target, copy."""
        context = encode_days(self.encoder, dynamic[:, :-1], static)[-1]  # (B, T, C, h, w)
        forecast = F.avg_pool2d(dynamic[:, -1, list(FORECAST_CHANNELS)], STRIDE, STRIDE)
        with torch.no_grad():
            target_days = encode_days(self.target_encoder, dynamic[:, -2:], static)[-1]  # days t and t + 1
            target, copy_last = channel_layer_norm(target_days[:, 1]), channel_layer_norm(target_days[:, 0])
        prediction = self.predict(context, forecast, copy_last)
        return {"context": context, "prediction": prediction, "target": target, "copy_last": copy_last}

    def predict(self, context: torch.Tensor, forecast: torch.Tensor, copy_last: torch.Tensor) -> torch.Tensor:
        """Predicted normalised latent of day t + 1; ``copy_last`` is the target encoder's normalised latent of t."""
        if self.target_mode == "change":
            return copy_last + self.predictor.change(context, forecast)
        return self.predictor(context, forecast)

    def loss(self, dynamic: torch.Tensor, static: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        z = self.latents(dynamic, static)
        fire_today = dynamic[:, -2, FIRE_CHANNEL] > 0.5
        fire_tomorrow = dynamic[:, -1, FIRE_CHANNEL] > 0.5
        near = near_fire_cells(fire_today, fire_tomorrow)
        emphasised = near if self.weighting == "near_fire" else change_cells(fire_today, fire_tomorrow)
        weight = 1 + (self.near_fire_weight - 1) * emphasised.float()
        error = ((z["prediction"] - z["target"]) ** 2).mean(1)  # (B, h, w)
        prediction_loss = (weight * error).sum() / weight.sum()
        if self.target_mode == "change":  # relative to copying today: shrinking day-to-day change does not help
            copy_error_now = ((z["copy_last"] - z["target"]) ** 2).mean(1)
            prediction_loss = prediction_loss / ((weight * copy_error_now).sum() / weight.sum()).clamp(min=1e-6)
        cells = z["context"][:, -1].movedim(1, -1).flatten(0, 2)  # today's context latents as (N, C)
        variance, covariance = variance_covariance(cells.float())
        total = prediction_loss + self.variance_weight * variance + self.covariance_weight * covariance
        with torch.no_grad():
            copy_error = ((z["copy_last"] - z["target"]) ** 2).mean(1)
            stats = {"prediction": prediction_loss.item(), "variance": variance.item(), "covariance": covariance.item(),
                     "near_fire_cells": int(near.sum()), "emphasised_cells": int(emphasised.sum()),
                     "prediction_error_near_fire": float(error[near].mean()) if near.any() else math.nan,
                     "copy_error_near_fire": float(copy_error[near].mean()) if near.any() else math.nan}  # fmt: skip
        return total, stats


def momentum_schedule(step: int, total: int, start: float = 0.996, end: float = 1.0) -> float:
    """Cosine increase of the EMA momentum from ``start`` to ``end`` over training (as in BYOL / I-JEPA)."""
    return end - (end - start) * (math.cos(math.pi * min(step, total) / max(total, 1)) + 1) / 2
