"""Check the report's stated BCE + batch-global smoothed Dice formula against the real training loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from fireline.models.spread import SpreadModel, spread_loss


def test_equal_weight_formula_masks_padding_and_is_batch_global() -> None:
    logits = torch.tensor([[[0.0, 1.0]], [[-1.0, 3.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    valid = torch.tensor([[[True, False]], [[True, True]]])
    p, y = torch.sigmoid(logits)[valid], target[valid]
    bce = -(y * torch.log(p) + (1 - y) * torch.log1p(-p)).mean()
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    actual = spread_loss(logits, target, valid)
    torch.testing.assert_close(actual, bce + dice)
    assert actual.dtype == torch.float32
    actual.backward()
    assert logits.grad is not None and logits.grad[~valid].eq(0).all()
    assert logits.grad[valid].abs().sum() > 0


def test_dice_ignores_true_negatives_but_counts_false_alarms() -> None:
    logits, target = torch.tensor([[[2.0, -1.0]]]), torch.tensor([[[1.0, 0.0]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = spread_loss(logits, target, valid) - bce
    more_logits = torch.cat([logits, torch.full((1, 1, 100), -100.0)], dim=-1)
    more_target = torch.cat([target, torch.zeros((1, 1, 100))], dim=-1)
    more_bce = F.binary_cross_entropy_with_logits(more_logits, more_target)
    more_dice = spread_loss(more_logits, more_target, torch.ones_like(more_target, dtype=torch.bool)) - more_bce
    torch.testing.assert_close(dice, more_dice)
    assert more_bce < bce  # easy background dilutes mean BCE, not Dice
    alarm_logits = logits.clone()
    alarm_logits[0, 0, 1] = 2
    alarm_dice = spread_loss(alarm_logits, target, valid) - F.binary_cross_entropy_with_logits(alarm_logits, target)
    assert alarm_dice > dice
    empty = spread_loss(torch.full((1, 2, 2), -100.0), torch.zeros((1, 2, 2)), torch.ones((1, 2, 2), dtype=torch.bool))
    assert torch.isfinite(empty) and empty == 0  # +1 smoothing, no 0/0


def test_one_output_loss_reaches_the_shared_encoder_through_all_days() -> None:
    model = SpreadModel(history=5, widths=(8, 8, 16, 16))
    dynamic = torch.randn(1, 5, 18, 32, 32, requires_grad=True)
    static = torch.randn(1, 21, 32, 32)
    target = torch.zeros(1, 32, 32)
    target[:, 10:15, 10:15] = 1
    loss = spread_loss(model(dynamic, static), target, torch.ones_like(target, dtype=torch.bool))
    loss.backward()
    assert dynamic.grad is not None
    assert (dynamic.grad.abs().sum(dim=(2, 3, 4)) > 0).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
