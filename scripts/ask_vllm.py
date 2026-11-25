"""Run decide() on one input through the vLLM server.

Usage:
    ./serve-model.sh    # in another terminal
    python scripts/ask_vllm.py "I was charged twice" \
        -q "Which team owns this ticket?" -o billing -o bug -o account -o other
"""

import argparse
import json
from pathlib import Path

from typed_decisions.backends import VllmScorer
from typed_decisions.decide import decide

parser = argparse.ArgumentParser()
parser.add_argument("state", help="text to be judged")
parser.add_argument("-q", "--question", required=True)
parser.add_argument("-o", "--option", action="append", required=True, dest="options")
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    help="default: results/temperature.json if present, else 1.0",
)
args = parser.parse_args()

temp_path = Path("results/temperature.json")
if args.temperature is not None:
    temperature: float = args.temperature
elif temp_path.exists():
    temperature = json.loads(temp_path.read_text())["temperature"]
else:
    temperature = 1.0

d = decide(VllmScorer(), args.state, args.question, args.options, temperature)
print(f"-> {d.option}   (option mass {d.mass:.3f}, temperature {temperature:.2f})")
for name, p in sorted(zip(args.options, d.probs, strict=True), key=lambda x: -x[1]):
    print(f"  {p:6.3f}  {name}")
