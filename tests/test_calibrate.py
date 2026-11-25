import numpy as np

from typed_decisions.calibrate import ece, fit_temperature, nll, softmax


def _overconfident(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, n)
    z = rng.normal(size=(n, 4))
    z[np.arange(n), y] += 1.0  # weak signal ...
    return z * 8.0, y  # ... inflated into overconfident logits


def test_softmax_rows_sum_to_one() -> None:
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, 0.0]]), 2.0)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_fit_temperature_cools_overconfidence() -> None:
    z, y = _overconfident()
    t = fit_temperature(z, y)
    assert t > 1.0
    assert nll(softmax(z, t), y) < nll(softmax(z), y)
    assert ece(softmax(z, t), y) < ece(softmax(z), y)


def test_ece_zero_for_perfect_confident_predictions() -> None:
    p = np.eye(3)
    assert ece(p, np.array([0, 1, 2])) == 0.0
