"""Fit a temperature on train scores and report calibration on test scores.

Usage:
    python scripts/calibrate.py

Needs results/transformers_train.npz and results/transformers_test.npz
(see scripts/eval_transformers.py). Writes results/temperature.json.
"""

import json

import numpy as np

from typed_decisions.calibrate import ece, fit_temperature, nll, softmax

tr = np.load("results/transformers_train.npz")
te = np.load("results/transformers_test.npz")
ztr, ytr = tr["z"].astype(np.float64), tr["y"]
zte, yte = te["z"].astype(np.float64), te["y"]

t = fit_temperature(ztr, ytr)
print(f"temperature {t:.3f}  (fit on {len(ytr)} train examples)")
print(f"{'':12}{'ECE':>8}{'NLL':>8}   (test, {len(yte)} examples)")
for label, temp in (("raw", 1.0), ("calibrated", t)):
    p = softmax(zte, temp)
    print(f"{label:12}{ece(p, yte):8.3f}{nll(p, yte):8.3f}")
print(f"accuracy {np.mean(zte.argmax(1) == yte):.3f} (unchanged by temperature)")

with open("results/temperature.json", "w") as f:
    json.dump({"temperature": t}, f)
