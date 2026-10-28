"""Single forward pass that returns scores for the option letters."""

import torch


@torch.no_grad()  # inference only; no gradient graph
def option_scores(model, tokenizer, prompt, ids):
    """Score the option letters at the first answer position.

    The model produces logits over the whole vocabulary at every position. Only
    the last position is used, since it predicts the first token of the answer.

    Args:
        model: Causal language model.
        tokenizer: Matching tokenizer.
        prompt: String from build_prompt.
        ids: Option token ids from option_ids.

    Returns:
        z: Tensor of shape (n_options,) with the raw option logits.
        mass: Combined probability the model assigns to the option letters,
            before renormalising. Values well below 1 mean the prompt is not
            producing a letter.
        top: The five highest-scoring tokens.
    """
    # add_special_tokens=False: the chat template already inserted any BOS token
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    logits = (
        model(**enc).logits[0, -1].float()
    )  # last position, over the full vocabulary
    logp = torch.log_softmax(logits, dim=-1)
    mass = logp[ids].exp().sum().item()  # total probability of the option letters
    z = logits[ids].cpu()  # raw scores for the options only
    top = [tokenizer.decode(i) for i in torch.topk(logits, 5).indices]
    return z, mass, top
