"""Next-day spread model: a per-day encoder shared across days, temporal fusion, and a U-Net decoder.

The encoder sees one day at a time (dynamic features of that day + static features) and returns feature maps at
strides 1, 2, 4 and 8. The stride-8 map is the latent grid a JEPA would predict. Fusion combines the ``history``
days at every scale with a 1x1 convolution over the stacked days; the decoder upsamples with skip connections and
receives the raw last-day fire mask at full resolution before the output layer (fire fronts are thin and sparse).
The supervised baseline trains everything; the JEPA arm reuses the same model with a pretrained, frozen encoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fireline.data.inputs import DYNAMIC_FEATURES, STATIC_FEATURES

FIRE_CHANNEL = DYNAMIC_FEATURES.index("active_fire")


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


class ConvBlock(nn.Module):
    """Two 3x3 conv + GroupNorm + GELU layers with a residual connection; optional stride-2 first conv."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), _norm(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), _norm(cout),
        )  # fmt: skip
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), _norm(cout))  # fmt: skip
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class DayEncoder(nn.Module):
    """One day (dynamic + static channels) -> features at strides 1, 2, 4, 8."""

    def __init__(self, in_channels: int = len(DYNAMIC_FEATURES) + len(STATIC_FEATURES),
                 widths: tuple[int, int, int, int] = (32, 64, 128, 128)) -> None:  # fmt: skip
        super().__init__()
        self.widths = widths
        self.stages = nn.ModuleList(
            [ConvBlock(in_channels, widths[0])]
            + [ConvBlock(widths[i - 1], widths[i], stride=2) for i in range(1, len(widths))]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def encode_days(encoder: DayEncoder, dynamic: torch.Tensor, static: torch.Tensor) -> list[torch.Tensor]:
    """Encode every day of ``dynamic`` (B, T, Cd, H, W) with ``static`` (B, Cs, H, W); returns (B, T, C, h, w) maps."""
    b, t = dynamic.shape[:2]
    days = torch.cat([dynamic, static[:, None].expand(-1, t, -1, -1, -1)], dim=2).flatten(0, 1)
    return [f.unflatten(0, (b, t)) for f in encoder(days)]


class SpreadModel(nn.Module):
    def __init__(self, history: int = 5, widths: tuple[int, int, int, int] = (32, 64, 128, 128),
                 extra_channels: int = 0) -> None:  # fmt: skip
        """``extra_channels``: per-day channels appended after the standard dynamic features (e.g. wind features)."""
        super().__init__()
        self.history = history
        self.extra_channels = extra_channels
        self.encoder = DayEncoder(
            in_channels=len(DYNAMIC_FEATURES) + extra_channels + len(STATIC_FEATURES), widths=widths
        )
        self.fuse = nn.ModuleList([nn.Sequential(nn.Conv2d(history * w, w, 1), _norm(w), nn.GELU()) for w in widths])
        self.up = nn.ModuleList([ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in range(len(widths) - 1)])
        self.head = nn.Sequential(ConvBlock(widths[0] + 1, widths[0]), nn.Conv2d(widths[0], 1, 1))

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Logits (B, H, W) of tomorrow's fire. ``dynamic`` (B, T, Cd, H, W), ``static`` (B, Cs, H, W)."""
        if dynamic.shape[1] != self.history:
            raise ValueError(f"Expected {self.history} days, got {dynamic.shape[1]}.")
        last_fire = dynamic[:, -1, FIRE_CHANNEL : FIRE_CHANNEL + 1]
        return self.decode(encode_days(self.encoder, dynamic, static), last_fire)

    def decode(self, encoded: list[torch.Tensor], last_fire: torch.Tensor) -> torch.Tensor:
        """Logits from per-day encoder features ``encoded`` (each (B, T, C, h, w)) and the last day's fire (B, 1, H, W).

        Lets evaluation encode every day of a fire once and reuse the features for all samples containing that day.
        """
        fused = [fuse(f.flatten(1, 2)) for fuse, f in zip(self.fuse, encoded, strict=True)]
        x = fused[-1]
        for i in range(len(fused) - 2, -1, -1):
            x = F.interpolate(x, size=fused[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.up[i](torch.cat([x, fused[i]], dim=1))
        return self.head(torch.cat([x, last_fire], dim=1))[:, 0]


def spread_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, dice_weight: float = 1.0
) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice over valid pixels (Dice counters the extreme class imbalance)."""
    valid = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / valid.sum().clamp(
        min=1
    )
    prob = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + (target * valid).sum() + 1)
    return bce + dice_weight * dice
