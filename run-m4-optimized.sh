#!/bin/bash
# Optimized llama.cpp run script for Apple M4 Mac Mini
# Based on benchmarking: flash attention + 8 threads gives best throughput

MODEL="${1:-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
shift 2>/dev/null

exec ./build/bin/llama-cli \
    -m "$MODEL" \
    -t 8 \
    --flash-attn on \
    -c 4096 \
    -ngl 99 \
    "$@"
