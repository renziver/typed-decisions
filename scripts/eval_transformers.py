"""Score n examples with the plain-transformers readout and report accuracy.

Usage:
    python scripts/eval_transformers.py 200 test      # evaluation numbers
    python scripts/eval_transformers.py 1000 train    # calibration scores (level 4)

The raw option scores are saved to results/transformers_<split>.npz for reuse
by the calibration and cascade code.
"""

import random
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from typed_decisions.data import load_banking77_subset
from typed_decisions.prompt import build_prompt, option_ids
from typed_decisions.readout import option_scores

NAME = "Qwen/Qwen3-1.7B"
QUESTION = "Which banking intent does this customer message express?"

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # number of examples to score
split = sys.argv[2] if len(sys.argv) > 2 else "test"  # "test" or "train"

train, test, options = load_banking77_subset()
data = list(test if split == "test" else train)
random.Random(0).shuffle(data)  # fixed shuffle; a raw split can be ordered by label

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, dtype=torch.bfloat16, device_map="cuda"
).eval()
ids = option_ids(tok, len(options))  # token id of each option letter

Z, M, Y = [], [], []  # option scores, option mass, true labels
for ex in tqdm(data[:n]):
    prompt = build_prompt(tok, ex["text"], QUESTION, options)
    z, mass, _ = option_scores(model, tok, prompt, ids)
    Z.append(z.numpy())
    M.append(mass)
    Y.append(ex["label"])

Z, M, Y = np.stack(Z), np.array(M), np.array(Y)  # shapes (n, k), (n,), (n,)
print(f"accuracy {np.mean(Z.argmax(1) == Y):.3f}   chance {1 / len(options):.3f}")
print(
    f"mean option mass {M.mean():.3f}   share of prompts under 0.9: {(M < 0.9).mean():.2f}"
)
np.savez(f"results/transformers_{split}.npz", z=Z, mass=M, y=Y)
