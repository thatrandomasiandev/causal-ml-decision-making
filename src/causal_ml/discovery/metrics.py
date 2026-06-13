"""Graph recovery metrics."""

from __future__ import annotations

import numpy as np


def _binarize(adj: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    return (np.abs(adj) > threshold).astype(int)


def structural_hamming_distance(true_adj: np.ndarray, pred_adj: np.ndarray, threshold: float = 0.1) -> int:
    """SHD: edge additions, deletions, and reversals."""
    true_bin = _binarize(true_adj, threshold)
    pred_bin = _binarize(pred_adj, threshold)
    d = true_bin.shape[0]
    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            t_ij, t_ji = true_bin[i, j], true_bin[j, i]
            p_ij, p_ji = pred_bin[i, j], pred_bin[j, i]
            if t_ij != p_ij or t_ji != p_ji:
                if (t_ij, t_ji) == (p_ji, p_ij) and (t_ij or t_ji):
                    shd += 1  # reversal
                else:
                    shd += abs(t_ij - p_ij) + abs(t_ji - p_ji)
    return int(shd)


def edge_precision_recall(
    true_adj: np.ndarray, pred_adj: np.ndarray, threshold: float = 0.1
) -> dict[str, float]:
    """Precision and recall for directed edge detection."""
    true_bin = _binarize(true_adj, threshold)
    pred_bin = _binarize(pred_adj, threshold)
    tp = int(np.sum(true_bin & pred_bin))
    fp = int(np.sum((1 - true_bin) & pred_bin))
    fn = int(np.sum(true_bin & (1 - pred_bin)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def orientation_accuracy(true_adj: np.ndarray, pred_adj: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of correctly oriented edges among recovered skeleton edges."""
    true_bin = _binarize(true_adj, threshold)
    pred_bin = _binarize(pred_adj, threshold)
    skeleton = (true_bin + true_bin.T + pred_bin + pred_bin.T) > 0
    correct = 0
    total = 0
    d = true_adj.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            if skeleton[i, j]:
                total += 1
                if true_bin[i, j] == pred_bin[i, j] and true_bin[j, i] == pred_bin[j, i]:
                    correct += 1
    return correct / total if total > 0 else 0.0
