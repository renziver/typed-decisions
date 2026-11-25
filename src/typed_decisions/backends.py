"""Backends that turn a question into option scores.

Both return scores that differ only by a per-prompt constant (raw logits vs
log-probabilities), so a softmax over the options gives the same probabilities.
"""

from typing import Protocol

import httpx
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from typed_decisions.prompt import build_prompt, option_ids
from typed_decisions.readout import option_scores

MODEL = "Qwen/Qwen3-1.7B"


class Scorer(Protocol):
    def score(
        self, state: str, question: str, options: list[str]
    ) -> tuple[np.ndarray, float]:
        """Return option scores of shape (n_options,) and the option mass."""
        ...


class TransformersScorer:
    """Runs the model in-process with transformers."""

    def __init__(self, name: str = MODEL) -> None:
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda"
        ).eval()  # type: ignore[no-untyped-call]

    def score(
        self, state: str, question: str, options: list[str]
    ) -> tuple[np.ndarray, float]:
        prompt = build_prompt(self.tok, state, question, options)
        z, mass, _ = option_scores(
            self.model, self.tok, prompt, option_ids(self.tok, len(options))
        )
        return z.numpy(), mass


class VllmScorer:
    """Reads option log-probabilities from a running vLLM server (serve-model.sh)."""

    TOP_LOGPROBS = 100  # must not exceed --max-logprobs in serve-model.sh

    def __init__(
        self, name: str = MODEL, url: str = "http://localhost:8000/v1/completions"
    ) -> None:
        self.name = name
        self.url = url
        self.tok = AutoTokenizer.from_pretrained(name)
        self.client = httpx.Client(timeout=60.0)

    def score(
        self, state: str, question: str, options: list[str]
    ) -> tuple[np.ndarray, float]:
        prompt = build_prompt(self.tok, state, question, options)
        resp = self.client.post(
            self.url,
            json={
                "model": self.name,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": self.TOP_LOGPROBS,
            },
        )
        resp.raise_for_status()
        top: dict[str, float] = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        letters = [str(self.tok.decode(i)) for i in option_ids(self.tok, len(options))]
        # A letter outside the top 100 is no likelier than the lowest one returned
        floor = min(top.values())
        z = np.array([top.get(c, floor) for c in letters], dtype=np.float64)
        mass = float(sum(np.exp(top[c]) for c in letters if c in top))
        return z, mass
