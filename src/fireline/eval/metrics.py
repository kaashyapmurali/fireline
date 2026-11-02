"""Streaming pixel metrics for next-day fire prediction, pooled over all evaluated pixels.

AP is computed from score histograms (``n_bins`` equal bins on [0, 1]); scores within one bin count as tied, which is
exact for binary scores and accurate to the bin width otherwise. Threshold metrics use exact counts at ``threshold``.
"""

from __future__ import annotations

import numpy as np


class PixelMetrics:
    def __init__(self, n_bins: int = 10_000, threshold: float = 0.5) -> None:
        self.n_bins, self.threshold = n_bins, threshold
        self.positives = np.zeros(n_bins, dtype=np.int64)
        self.negatives = np.zeros(n_bins, dtype=np.int64)
        self.tp = self.fp = self.fn = 0
        self.squared_error = 0.0

    def update(self, scores: np.ndarray, targets: np.ndarray, valid: np.ndarray | None = None) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        targets = np.asarray(targets).astype(bool)
        if valid is not None:
            scores, targets = scores[valid], targets[valid]
        scores, targets = scores.ravel(), targets.ravel()
        if np.any((scores < 0) | (scores > 1)) or np.any(~np.isfinite(scores)):
            raise ValueError("Scores must be finite probabilities in [0, 1].")
        bins = np.minimum((scores * self.n_bins).astype(np.int64), self.n_bins - 1)
        self.positives += np.bincount(bins[targets], minlength=self.n_bins)
        self.negatives += np.bincount(bins[~targets], minlength=self.n_bins)
        flagged = scores >= self.threshold
        self.tp += int((flagged & targets).sum())
        self.fp += int((flagged & ~targets).sum())
        self.fn += int((~flagged & targets).sum())
        self.squared_error += float(((scores - targets) ** 2).sum())

    def average_precision(self) -> float | None:
        total_pos = int(self.positives.sum())
        if total_pos == 0:
            return None
        tp = np.cumsum(self.positives[::-1])  # thresholds from high to low
        fp = np.cumsum(self.negatives[::-1])
        flagged = tp + fp
        precision = np.divide(tp, flagged, out=np.zeros(len(tp)), where=flagged > 0)
        recall = tp / total_pos
        return float(np.sum(np.diff(recall, prepend=0.0) * precision))

    def compute(self) -> dict:
        n_pos = int(self.positives.sum())
        n = n_pos + int(self.negatives.sum())
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        recall = self.tp / n_pos if n_pos else None
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else None
        iou = self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else None
        return {
            "ap": self.average_precision(),
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "flagged_over_observed": (self.tp + self.fp) / n_pos if n_pos else None,
            "brier": self.squared_error / n if n else None,
            "prevalence": n_pos / n if n else None,
            "pixels": n,
            "positive_pixels": n_pos,
            "threshold": self.threshold,
        }
