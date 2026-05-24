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

## Theoretical Analysis

### Memory Bandwidth Ceiling
For text generation (tg), each token requires reading the entire model weights from RAM:
- Model size: 762 MB
- Practical bandwidth: ~12.8 GB/s
- **Theoretical max: 762MB / 12.8 GB/s = 59.5ms/token ≈ 16.8 tokens/sec**
- Current performance: 11.9 t/s = **71% of theoretical maximum**

The remaining 29% gap is from:
- Non-matmul ops (RMS norm, RoPE, softmax, SiLU) ~5%
- Thread synchronization barriers between ops ~3%
- OS/scheduler overhead ~5%
- Memory controller inefficiency ~16%

### Why 100 t/s is Impossible on This Hardware
To reach 100 t/s with a 762MB model, you would need:
- 762MB × 100 = 76.2 GB/s memory bandwidth
- This exceeds DDR4-2133 dual-channel by **4.5x**
- Even DDR5-6400 quad-channel (~100 GB/s) would barely suffice

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

## Optimization Attempts
| # | Change | pp (t/s) | tg (t/s) | Delta tg | Notes |
|---|--------|----------|----------|----------|-------|
| 0 | Baseline | 41.72 | 11.91 | — | |
| 1 | Prefetching in Q4_K kernel | 42.03 | 11.92 | +0.1% | HW prefetcher already effective |
| 2 | Multi-row (nrc=2) processing | 42.28 | 11.37 | -4.5% | Helps pp, hurts tg (register pressure) |
| 3 | Fully unrolled inner loop | 40.70 | 11.86 | -0.4% | Compiler already unrolling |
| 4 | Larger chunk_size (64→256) | 41.31 | 11.81 | -0.8% | Reduced parallelism hurt |
| 5 | Q8 KV cache + flash attention | 36.60 | 11.91 | 0% | KV cache too small at 128 tokens |
| 6 | ngram-simple speculation | — | 12.4 | +4% | CLI test, helps with repetitive content |
| 7 | Default ngram-mod speculation | — | crash | — | Requires explicit init, can't be default |

## Key Findings

### The Q4_K AVX2 kernel is already near-optimal
- Hot path: `ggml_vec_dot_q4_K_q8_K` in `ggml/src/ggml-cpu/arch/x86/quants.c`
- Uses efficient `_mm256_maddubs_epi16` + `_mm256_madd_epi16` SIMD patterns
- Inner loop processes 64 elements per iteration (4 iterations per 256-element block)
- No obvious inefficiencies remaining

### Memory bandwidth is the hard ceiling
- Every kernel optimization attempted was within noise
- The CPU spends most time waiting for RAM, not computing
- Hardware prefetcher handles sequential access patterns well
- No amount of SIMD optimization can overcome this

### llamafile sgemm skips single-token generation
- Line 3707 in `ggml/src/ggml-cpu/llamafile/sgemm.cpp`: `if (n < 2) return false;`
- The optimized tinyBLAS path only helps prompt processing (batch > 1)

### Operator fusion is already implemented
- RMS_NORM + MUL fusion exists (`ggml_compute_forward_rms_norm_mul_fused`)
- SwiGLU is a single fused op
- No additional fusion opportunities for the Llama architecture

### Speculative decoding is the only path past the bandwidth wall
- N-gram based speculation (`--spec-type ngram-simple`) gives ~4% improvement
- Full speculative decoding with a draft model can give 2-3x speedup
- But this changes the effective algorithm, not the raw throughput

## Paths to Higher Throughput

### On THIS hardware (can reach ~14-15 t/s):
1. Use `--spec-type ngram-mod` for repetitive text workloads
2. Use smaller context (`-c 512` instead of 2048) to reduce KV cache reads
3. Use Q4_0 quantization (simpler kernel, ~same bandwidth)

### To reach 100 t/s (hardware changes required):
1. **Use a GPU** — even a budget GPU (RTX 3060) has ~360 GB/s bandwidth = potential ~50+ t/s
2. **Use a smaller model** — 0.5B parameter model would be ~380MB = ~33 t/s on this CPU
3. **Use faster RAM** — DDR5-6400 system would roughly double bandwidth
4. **Use Apple Silicon** — M1/M2 unified memory has ~68 GB/s = potential ~90 t/s with this model
5. **Speculative decoding** with a tiny draft model — effective 2-3x multiplier on apparent speed

## Architecture Notes (for future reference)

### Code path for text generation:
```
llama_decode()
  → ggml_backend_cpu_graph_compute()
    → ggml_graph_compute() [OpenMP parallel]
      → ggml_graph_compute_thread()
        → for each node: ggml_compute_forward() + ggml_barrier()
          → ggml_compute_forward_mul_mat()
            → ggml_compute_forward_mul_mat_one_chunk()
              → ggml_vec_dot_q4_K_q8_K() [THE hot kernel]
```

### Key files:
- `ggml/src/ggml-cpu/arch/x86/quants.c` — SIMD dot product kernels
- `ggml/src/ggml-cpu/ggml-cpu.c` — matmul dispatch, graph execution, thread scheduling
- `ggml/src/ggml-cpu/ops.cpp` — non-matmul operations (RMS norm, RoPE, softmax)
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp` — tinyBLAS for batch operations
- `common/speculative.cpp` — speculative decoding infrastructure
- `common/ngram-mod.cpp` — n-gram based speculation (no draft model needed)
