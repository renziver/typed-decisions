"""Dataset loading."""

import random
from typing import TypedDict

from datasets import load_dataset


class Example(TypedDict):
    text: str
    label: int


def load_banking77_subset(
    k: int = 20, seed: int = 0
) -> tuple[list[Example], list[Example], list[str]]:
    """Load Banking77 restricted to k intents.

    Options are labelled with single letters, which limits a question to 26
    classes. A seeded random sample of k intents is kept and all other
    examples are dropped.

    Args:
        k: Number of intents to keep (at most 26).
        seed: Seed used to choose the intents.

    Returns:
        train: List of Example. Used for prompt design and
            calibration; no model is trained on it.
        test: Same format. Used for final numbers only.
        options: Intent names in label order, so options[label] is the name.
    """
    ds = load_dataset("mteb/banking77")  # parquet mirror; PolyAI's is script-based
    id_to_name = {r["label"]: r["label_text"] for r in ds["train"]}
    names = [id_to_name[i] for i in range(len(id_to_name))]  # all 77 intent names
    keep = sorted(
        random.Random(seed).sample(range(len(names)), k)
    )  # indices of the kept intents
    remap = {old: new for new, old in enumerate(keep)}  # original label id -> 0..k-1
    options = [names[i].replace("_", " ") for i in keep]

    def pick(split: str) -> list[Example]:
        """Return the split's examples with labels remapped to 0..k-1."""
        return [
            {"text": r["text"], "label": remap[r["label"]]}
            for r in ds[split]
            if r["label"] in remap
        ]

    return pick("train"), pick("test"), options
