"""Typed decision: one question in, one option plus probabilities out."""

from dataclasses import dataclass

import numpy as np

from typed_decisions.backends import Scorer


@dataclass(frozen=True)
class Decision:
    probs: list[float]  # one probability per option, summing to 1
    choice: int  # index of the most probable option
    option: str  # name of the most probable option
    mass: float  # probability the model put on the option letters, before renormalising


def decide(
    scorer: Scorer,
    state: str,
    question: str,
    options: list[str],
    temperature: float = 1.0,
) -> Decision:
    """Pick one of the options for the text.

    Args:
        scorer: Backend that scores the options.
        state: Text to be judged.
        question: Instruction shown to the model.
        options: Option names (at most 26).
        temperature: Fitted by scripts/calibrate.py; 1.0 leaves scores unchanged.
    """
    z, mass = scorer.score(state, question, options)
    s = z / temperature
    p = np.exp(s - s.max())
    p /= p.sum()
    choice = int(p.argmax())
    return Decision(probs=p.tolist(), choice=choice, option=options[choice], mass=mass)
