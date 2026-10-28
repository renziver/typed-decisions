# typed-decisions

Get a typed decision out of a small LLM with a single forward pass. A multiple-choice question is shown with lettered options, and the logits of the option letters at the first answer position are read directly. This gives a probability for every option instead of free text that has to be parsed.

The test case is [Banking77](https://huggingface.co/datasets/mteb/banking77) restricted to 20 intents, scored with `Qwen/Qwen3-1.7B`.

## Setup

Requires Python 3.14, [uv](https://docs.astral.sh/uv/) and a CUDA GPU.

```bash
uv sync
```

## Usage

Run the readout once on a ticket-routing question:

```bash
uv run python src/typed_decisions/test.py
```

Score a number of examples and report accuracy:

```bash
uv run python scripts/eval_transformers.py 200 test
```

The raw option scores are saved to `results/transformers_<split>.npz` (`z` scores, `mass` option mass, `y` labels).

`mass` is the total probability the model puts on the option letters before renormalising. Values well below 1 mean the prompt is not producing a letter.

## Layout

- `src/typed_decisions/data.py`: Banking77 loading and intent subsetting
- `src/typed_decisions/prompt.py`: chat prompt that ends where the answer begins
- `src/typed_decisions/readout.py`: single forward pass returning option scores
- `scripts/eval_transformers.py`: accuracy evaluation

## Status

- [x] Level 0: environment
- [x] Level 1: plain transformers readout
- [x] Level 2: data and accuracy
- [ ] Level 3: vLLM serving and parity
- [ ] Level 4: calibration
- [ ] Level 5: question types and API surface
- [ ] Level 6: experiments
- [ ] Level 7: write-up
