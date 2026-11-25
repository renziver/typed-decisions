#!/usr/bin/env bash
# FlashInfer's sampler JIT-compiles with nvcc, which isn't installed; we only read logprobs
export VLLM_USE_FLASHINFER_SAMPLER=0
uv run --project "$(dirname "$0")/vllm-server" vllm serve "Qwen/Qwen3-1.7B" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-logprobs 100
