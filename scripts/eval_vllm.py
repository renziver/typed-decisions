"""Score the same examples through the vLLM server and compare with transformers.

Usage:
    ./serve-model.sh                              # in another terminal
    python scripts/eval_vllm.py 500 test

Saves results/vllm_<split>.npz. If results/transformers_<split>.npz holds at
least n examples, reports parity against it.
"""

import random
import sys
from pathlib import Path

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

ref_path = Path(f"results/transformers_{split}.npz")
if ref_path.exists():
    ref = np.load(ref_path)
    if len(ref["y"]) >= n:
        zr = ref["z"][:n].astype(np.float64)
        assert (ref["y"][:n] == Y).all(), "example order differs"

        def logsoftmax(a: np.ndarray) -> np.ndarray:
            a = a - a.max(1, keepdims=True)
            return a - np.log(np.exp(a).sum(1, keepdims=True))

        diff = np.abs(np.exp(logsoftmax(zr)) - np.exp(logsoftmax(Z)))
        agree = np.mean(zr.argmax(1) == Z.argmax(1))
        print(f"argmax agreement {agree:.3f}")
        print(f"option probability gap: max {diff.max():.4f}   mean {diff.mean():.5f}")
