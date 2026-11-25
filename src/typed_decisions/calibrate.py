"""Temperature scaling and calibration metrics."""

from itertools import pairwise

import numpy as np


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of z / temperature."""
    s = z / temperature
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return np.asarray(e / e.sum(axis=1, keepdims=True))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true labels."""
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: gap between confidence and accuracy per bin."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in pairwise(edges):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(conf[in_bin].mean() - correct[in_bin].mean())
    return float(total)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Return the temperature that minimises NLL on (z, y), by grid search.

    A single scalar cannot overfit much, so a grid over log-temperature is
    enough and needs no optimiser.
    """
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    losses = [nll(softmax(z, t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])
