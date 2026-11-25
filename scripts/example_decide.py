"""Two question types through the same decide() call: multi-class and yes/no.

Usage:
    python scripts/example_decide.py
"""

import json
from pathlib import Path

from typed_decisions.backends import TransformersScorer
from typed_decisions.decide import decide

path = Path("results/temperature.json")
temperature = json.loads(path.read_text())["temperature"] if path.exists() else 1.0

scorer = TransformersScorer()

routing = decide(
    scorer,
    "I was charged twice for my subscription this month.",
    "Which team owns this ticket?",
    ["billing", "bug", "account", "other"],
    temperature,
)
print(routing.option, [round(p, 3) for p in routing.probs])

refund = decide(
    scorer,
    "The app crashed and I lost my order, I want my money back.",
    "Is the customer asking for a refund?",
    ["yes", "no"],
    temperature,
)
print(refund.option, [round(p, 3) for p in refund.probs])
