"""Score examples through the vLLM server and report accuracy.

Usage:
    ./serve-model.sh                              # in another terminal
    python scripts/eval_vllm.py 500 test

Saves the raw option scores to results/vllm_<split>.npz.
"""

import random
import sys

import numpy as np
from tqdm import tqdm

from typed_decisions.backends import VllmScorer
from typed_decisions.data import load_banking77_subset

QUESTION = "Which banking intent does this customer message express?"

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
split = sys.argv[2] if len(sys.argv) > 2 else "test"

train, test, options = load_banking77_subset()
data = list(test if split == "test" else train)
random.Random(0).shuffle(data)  # same shuffle as eval_transformers.py

scorer = VllmScorer()
scores: list[np.ndarray] = []
masses: list[float] = []
labels: list[int] = []
for ex in tqdm(data[:n]):
    z, mass = scorer.score(ex["text"], QUESTION, options)
    scores.append(z)
    masses.append(mass)
    labels.append(ex["label"])

Z, M, Y = np.stack(scores), np.array(masses), np.array(labels)
print(f"accuracy {np.mean(Z.argmax(1) == Y):.3f}   mean option mass {M.mean():.3f}")
np.savez(f"results/vllm_{split}.npz", z=Z, mass=M, y=Y)
