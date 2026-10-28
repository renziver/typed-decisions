# Run the readout once on a ticket-routing question.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from typed_decisions.prompt import build_prompt, option_ids
from typed_decisions.readout import option_scores

name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    dtype=torch.bfloat16,
    device_map="cuda",  # older transformers versions call this torch_dtype
).eval()  # type: ignore[no-untyped-call]

options = ["billing", "bug", "account", "other"]
prompt = build_prompt(
    tok,
    "There is a bug",
    "Which team owns this ticket?",
    options,
)
z, mass, top = option_scores(model, tok, prompt, option_ids(tok, len(options)))

print(torch.softmax(z, dim=-1))  # probability per option
print(mass)  # combined probability of the option letters
print(top)  # five highest-scoring tokens
