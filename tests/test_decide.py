import numpy as np

from typed_decisions.decide import decide


class FakeScorer:
    def score(
        self, state: str, question: str, options: list[str]
    ) -> tuple[np.ndarray, float]:
        return np.array([0.0, 2.0, 1.0]), 0.97


def test_decide_picks_highest_score() -> None:
    d = decide(FakeScorer(), "text", "q?", ["a", "b", "c"])
    assert (d.choice, d.option, d.mass) == (1, "b", 0.97)
    assert abs(sum(d.probs) - 1.0) < 1e-9


def test_temperature_flattens_probabilities() -> None:
    sharp = decide(FakeScorer(), "t", "q?", ["a", "b", "c"], temperature=1.0)
    flat = decide(FakeScorer(), "t", "q?", ["a", "b", "c"], temperature=5.0)
    assert max(flat.probs) < max(sharp.probs)
    assert flat.choice == sharp.choice
