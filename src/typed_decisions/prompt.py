"""Prompt construction for the one-token readout.

The prompt must end where the assistant's answer would begin, so that the next
token the model predicts is an option letter.
"""

from transformers import PreTrainedTokenizerBase

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # one token per label, so at most 26 options


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    state: str,
    question: str,
    options: list[str],
) -> str:
    """Build the chat prompt for one question.

    Args:
        tokenizer: Hugging Face tokenizer with a chat template.
        state: Text to be judged, for example a support ticket.
        question: Instruction shown to the model.
        options: Option names. Option i is labelled LETTERS[i].

    Returns:
        Prompt string that ends at the start of the assistant turn.
    """
    lines = [
        f"{LETTERS[i]}) {o}" for i, o in enumerate(options)
    ]  # "A) billing", "B) bug", ...
    user = (
        f"{question}\n\n"
        + "\n".join(lines)
        + f"\n\nText: {state}\n\nAnswer with the letter only."
    )
    messages = [{"role": "user", "content": user}]
    # enable_thinking is read by Qwen3's template and ignored by others
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert isinstance(prompt, str)
    return prompt


def option_ids(tokenizer: PreTrainedTokenizerBase, n: int) -> list[int]:
    """Return the token id of each option letter.

    Args:
        tokenizer: Tokenizer used with build_prompt.
        n: Number of options (at most 26).

    Returns:
        List of n token ids, in option order.

    Raises:
        AssertionError: If a letter does not encode to exactly one token.
    """
    ids = []
    for i in range(n):
        t = tokenizer.encode(LETTERS[i], add_special_tokens=False)
        assert len(t) == 1, f"{LETTERS[i]} is not a single token"
        ids.append(t[0])
    return ids
