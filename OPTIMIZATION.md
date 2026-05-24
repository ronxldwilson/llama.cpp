# CPU Inference Optimization Project

## Goal
Maximize token generation speed on CPU-only, low-end hardware by making additive code changes to the llama.cpp inference engine.

## Hardware
| Spec | Value |
|------|-------|
| CPU | Intel Core i3-7020U @ 2.30GHz (fixed, no turbo) |
| Cores | 2 physical / 4 logical (hyperthreading) |
| RAM | 8 GB DDR4-2133 (dual channel) |
| GPU | None |
| SIMD | SSE4.2, AVX2, FMA |
| L1 Cache | 64KB per core |
| L2 Cache | 256KB per core |
| L3 Cache | 3MB shared |
| Memory Bandwidth | ~12.8 GB/s practical (~17 GB/s theoretical) |

## Model
- Llama 3.2 1B Instruct (Q4_K_M quantization)
- Size: 762.81 MiB
- Parameters: 1.24B

## Theoretical Limits
- **tg ceiling**: 762MB model / 12.8 GB/s bandwidth ≈ **~16.8 tokens/sec**
- Current tg at 11.91 = **71% of theoretical max** (already well-optimized)
- Breaking past 17 t/s requires algorithmic changes (speculative decoding) not kernel tuning

## Baseline Benchmark (2026-05-25)
| Metric | Tokens/sec |
|--------|-----------|
| Prompt Processing (pp512) | 41.72 ± 2.65 |
| Text Generation (tg128) | 11.91 ± 0.23 |

- Build: 549b9d843 (commit 9307)
- Threads: 4
- Backend: CPU

## Thread Scaling
| Threads | pp (t/s) | tg (t/s) |
|---------|----------|----------|
| 1 | 18.91 | 7.14 |
| 2 | 30.88 | 10.73 |
| 3 | 40.53 | 11.03 |
| 4 | 41.72 | 11.91 |

tg scaling from 2→4 threads is only +10%, confirming memory bandwidth saturation.

## Optimization Log
| # | Change | pp (t/s) | tg (t/s) | Delta tg | Notes |
|---|--------|----------|----------|----------|-------|
| 0 | Baseline | 41.72 | 11.91 | — | |
| 1 | Prefetching in Q4_K kernel | 42.03 | 11.92 | +0.1% | Within noise — HW prefetcher already effective |
| 2 | Multi-row (nrc=2) processing | 42.28 | 11.37 | -4.5% | Helps pp but hurts tg (register pressure) |
| 3 | Fully unrolled inner loop | 40.70 | 11.86 | -0.4% | Compiler already unrolling; no benefit |
| 4 | Larger chunk_size (64→256) | 41.31 | 11.81 | -0.8% | Reduced parallelism hurt more than sync saved |
| 5 | Q8 KV cache + flash attention | 36.60 | 11.91 | 0% | KV cache too small at 128 tokens to matter |

## Key Findings
1. **The Q4_K AVX2 kernel is already near-optimal.** The hot path (`ggml_vec_dot_q4_K_q8_K`) uses efficient `_mm256_maddubs_epi16` + `_mm256_madd_epi16` patterns.
2. **Memory bandwidth is the hard ceiling.** Every kernel optimization attempted was within noise because the CPU spends most of its time waiting for RAM.
3. **llamafile sgemm skips n=1** (single token generation), so the tinyBLAS path doesn't help for tg.
4. **4 threads is optimal** for this 2-core/4-thread CPU.

## Next Steps: Algorithmic Changes
To break past the memory bandwidth wall, we need to generate more tokens per model forward pass:
1. **Speculative decoding** — use a draft model to predict multiple tokens, verify in one pass
2. **Prompt lookup decoding** — match n-grams from prompt to predict ahead (no draft model needed)
3. **Early exit / layer skipping** — skip less important layers during generation
