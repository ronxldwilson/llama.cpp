"""
SVD Analysis of Llama 3.2 1B FFN Weight Matrices

This script analyzes how much we can compress the FFN weight matrices
using low-rank approximation (truncated SVD) while measuring quality loss.

The goal: if we can approximate W ≈ A × B where A is [n_ff × r] and B is [r × n_embd],
with r << min(n_ff, n_embd), we read fewer bytes per token from RAM.

Memory savings per FFN matmul:
  Original: n_ff × n_embd values
  Factored: n_ff × r + r × n_embd values
  Ratio: (n_ff × r + r × n_embd) / (n_ff × n_embd) = r/n_embd + r/n_ff
"""

import sys
import struct
import numpy as np
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

# Q4_K block: 256 elements packed into 144 bytes
QK_K = 256
BLOCK_Q4_K_SIZE = 144  # 2(d) + 2(dmin) + 12(scales) + 128(qs)


def dequantize_q4_k_block(block_data):
    """Dequantize a single Q4_K block (256 values from 144 bytes)."""
    d = struct.unpack('<e', block_data[0:2])[0]       # fp16 scale
    dmin = struct.unpack('<e', block_data[2:4])[0]    # fp16 min
    scales = block_data[4:16]                          # 12 bytes of packed scales
    qs = block_data[16:144]                            # 128 bytes of quantized values

    # Unpack scales (6-bit) and mins (6-bit) from 12 bytes
    # First 8 groups: scales in lower 6 bits, mins in upper 6 bits (packed in first 8 bytes)
    # Last 4 groups: packed differently in remaining 4 bytes
    sc = np.zeros(8, dtype=np.float32)
    mn = np.zeros(8, dtype=np.float32)

    # Groups 0-3: scales from lower 6 bits of bytes 0-3, mins from lower 6 bits of bytes 4-7
    for i in range(4):
        sc[i] = (scales[i] & 0x3F)
        mn[i] = (scales[i + 4] & 0x3F)

    # Groups 4-7: scales and mins from bytes 8-11 combined with upper bits of 0-7
    for i in range(4):
        sc[i + 4] = ((scales[i + 8] & 0x0F) | ((scales[i] >> 6) << 4))
        mn[i + 4] = ((scales[i + 8] >> 4)   | ((scales[i + 4] >> 6) << 4))

    # Dequantize: each byte in qs holds two 4-bit values (low and high nibble)
    result = np.zeros(256, dtype=np.float32)
    for group in range(8):
        group_d = d * sc[group]
        group_m = dmin * mn[group]
        for j in range(32):
            byte_idx = group * 16 + j % 16
            if j < 16:
                val = qs[byte_idx] & 0x0F
            else:
                val = (qs[byte_idx] >> 4) & 0x0F
            result[group * 32 + j] = group_d * val - group_m

    return result


def dequantize_q4_k_tensor(data, n_rows, n_cols):
    """Dequantize a full Q4_K tensor to float32."""
    assert n_cols % QK_K == 0, f"n_cols ({n_cols}) must be multiple of {QK_K}"
    blocks_per_row = n_cols // QK_K

    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        for blk in range(blocks_per_row):
            offset = (row * blocks_per_row + blk) * BLOCK_Q4_K_SIZE
            block_data = data[offset:offset + BLOCK_Q4_K_SIZE]
            result[row, blk*QK_K:(blk+1)*QK_K] = dequantize_q4_k_block(block_data)

    return result


def read_gguf_tensor(model_path, tensor_name):
    """Read a specific tensor from a GGUF file."""
    with open(model_path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic: {hex(magic)})"

        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata entries")

        # Skip metadata key-value pairs
        for _ in range(n_kv):
            # Read key (string)
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')

            # Read value type
            vtype = struct.unpack('<I', f.read(4))[0]

            # Skip value based on type
            if vtype == 0:  # UINT8
                f.read(1)
            elif vtype == 1:  # INT8
                f.read(1)
            elif vtype == 2:  # UINT16
                f.read(2)
            elif vtype == 3:  # INT16
                f.read(2)
            elif vtype == 4:  # UINT32
                f.read(4)
            elif vtype == 5:  # INT32
                f.read(4)
            elif vtype == 6:  # FLOAT32
                f.read(4)
            elif vtype == 7:  # BOOL
                f.read(1)
            elif vtype == 8:  # STRING
                str_len = struct.unpack('<Q', f.read(8))[0]
                f.read(str_len)
            elif vtype == 9:  # ARRAY
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for _ in range(arr_len):
                    if arr_type == 0: f.read(1)
                    elif arr_type == 1: f.read(1)
                    elif arr_type == 2: f.read(2)
                    elif arr_type == 3: f.read(2)
                    elif arr_type == 4: f.read(4)
                    elif arr_type == 5: f.read(4)
                    elif arr_type == 6: f.read(4)
                    elif arr_type == 7: f.read(1)
                    elif arr_type == 8:
                        sl = struct.unpack('<Q', f.read(8))[0]
                        f.read(sl)
                    elif arr_type == 10: f.read(8)
                    elif arr_type == 11: f.read(8)
                    elif arr_type == 12: f.read(8)
                    else:
                        raise ValueError(f"Unknown array element type {arr_type}")
            elif vtype == 10:  # UINT64
                f.read(8)
            elif vtype == 11:  # INT64
                f.read(8)
            elif vtype == 12:  # FLOAT64
                f.read(8)
            else:
                raise ValueError(f"Unknown metadata type {vtype} for key '{key}'")

            # Print some interesting metadata
            if 'feed_forward' in key or 'embedding' in key or 'block_count' in key:
                # Re-read value for display
                pass

        # Read tensor info
        tensor_infos = {}
        for _ in range(n_tensors):
            # Tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')

            # Number of dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]

            # Shape
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack('<Q', f.read(8))[0])

            # Type
            dtype = struct.unpack('<I', f.read(4))[0]

            # Offset from start of data section
            offset = struct.unpack('<Q', f.read(8))[0]

            tensor_infos[name] = {
                'shape': shape,
                'type': dtype,
                'offset': offset,
                'n_dims': n_dims,
            }

        # Data section starts after alignment
        data_start = f.tell()
        alignment = 32  # GGUF default alignment
        data_start = ((data_start + alignment - 1) // alignment) * alignment

        # Find and read our tensor
        if tensor_name not in tensor_infos:
            print(f"Tensor '{tensor_name}' not found!")
            print(f"Available tensors with 'ffn': ")
            for name in sorted(tensor_infos.keys()):
                if 'ffn' in name:
                    info = tensor_infos[name]
                    print(f"  {name}: shape={info['shape']}, type={info['type']}")
            return None, None

        info = tensor_infos[tensor_name]
        print(f"Found tensor: {tensor_name}")
        print(f"  Shape: {info['shape']}")
        print(f"  Type: {info['type']} (Q4_K = 12)")

        # Read raw data
        n_elements = 1
        for s in info['shape']:
            n_elements *= s

        # For Q4_K: each block of 256 elements takes 144 bytes
        n_blocks = n_elements // QK_K
        data_size = n_blocks * BLOCK_Q4_K_SIZE

        f.seek(data_start + info['offset'])
        raw_data = f.read(data_size)

        return raw_data, info


def analyze_svd(matrix, name, ranks_to_test=None):
    """Perform SVD analysis on a matrix and report compression vs error."""
    n_rows, n_cols = matrix.shape

    if ranks_to_test is None:
        ranks_to_test = [32, 64, 128, 256, 512, 768, 1024, 1536]

    # Filter out ranks larger than min dimension
    max_rank = min(n_rows, n_cols)
    ranks_to_test = [r for r in ranks_to_test if r < max_rank]

    print(f"\n{'='*70}")
    print(f"SVD Analysis: {name} [{n_rows} x {n_cols}]")
    print(f"Original size: {n_rows * n_cols * 0.5 / 1024:.1f} KB (Q4_K)")
    print(f"Original size (f32): {n_rows * n_cols * 4 / 1024:.1f} KB")
    print(f"Frobenius norm: {np.linalg.norm(matrix):.4f}")
    print(f"{'='*70}")

    # Full SVD (truncated for efficiency — we only need top-k singular values)
    print("Computing SVD (this may take a minute)...")

    # Use randomized SVD for speed if matrix is large
    if max_rank > 1024:
        # Only compute top ranks we need
        max_needed = max(ranks_to_test)
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    else:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    print(f"Singular values range: {S[0]:.4f} (max) to {S[-1]:.6f} (min)")
    print(f"Top 10 singular values: {S[:10]}")
    print(f"Condition number: {S[0]/S[-1]:.1f}")

    # Analyze each rank
    print(f"\n{'Rank':<6} {'Size(KB)':<10} {'Compress':<10} {'RelError':<10} {'Bandwidth':<12} {'Est t/s':<8}")
    print(f"{'----':<6} {'--------':<10} {'--------':<10} {'--------':<10} {'--------':<12} {'------':<8}")

    original_norm = np.linalg.norm(matrix)
    original_bytes_q4k = n_rows * n_cols // QK_K * BLOCK_Q4_K_SIZE

    for rank in ranks_to_test:
        # Reconstruct at this rank
        reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

        # Error metrics
        error = np.linalg.norm(matrix - reconstructed)
        rel_error = error / original_norm

        # Size of factored form (stored as FP16 for A and B)
        # A = U[:, :rank] * sqrt(S[:rank]) → [n_rows × rank] × 2 bytes
        # B = sqrt(S[:rank]) * Vt[:rank, :] → [rank × n_cols] × 2 bytes
        factored_bytes = (n_rows * rank + rank * n_cols) * 2  # FP16
        compression = factored_bytes / original_bytes_q4k

        # Estimated speed: proportional to bytes read
        # Baseline: 762MB model → 11.9 t/s
        # If we replace FFN weights (which are ~80% of model), the new model size changes
        # There are 3 FFN matrices per layer × 16 layers = 48 matrices
        # Gate+Up are [n_embd, n_ff] and Down is [n_ff, n_embd]
        # Let's estimate for this one matrix type
        bandwidth_ratio = factored_bytes / original_bytes_q4k

        print(f"{rank:<6} {factored_bytes/1024:<10.1f} {compression:<10.2f}x {rel_error:<10.4f} {bandwidth_ratio:<12.2f}x {'GOOD' if compression < 1.0 else 'WORSE'}")

    # Find optimal rank (where compression < 1 and error < 5%)
    print(f"\n--- Recommendations ---")
    for rank in ranks_to_test:
        reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        rel_error = np.linalg.norm(matrix - reconstructed) / original_norm
        factored_bytes = (n_rows * rank + rank * n_cols) * 2
        compression = factored_bytes / original_bytes_q4k

        if compression < 1.0 and rel_error < 0.05:
            print(f"  Best: rank={rank}, compression={compression:.2f}x, error={rel_error:.4f}")
            break
    else:
        print("  No rank achieves both <1x compression AND <5% error with FP16 factors")
        print("  (Q4_K is already very compressed — hard to beat with SVD)")

    return S


def main():
    model_path = Path(r"F:\inference-engine\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Analyze layer 0's FFN gate weight as a representative
    tensor_name = "blk.0.ffn_gate.weight"

    raw_data, info = read_gguf_tensor(str(model_path), tensor_name)

    if raw_data is None:
        return

    # Dequantize to float32 for SVD analysis
    shape = info['shape']
    n_cols = shape[0]  # n_embd (inner dimension)
    n_rows = shape[1]  # n_ff (outer dimension)

    print(f"\nDequantizing {tensor_name} [{n_rows} x {n_cols}]...")
    matrix = dequantize_q4_k_tensor(raw_data, n_rows, n_cols)

    print(f"Matrix stats: mean={matrix.mean():.6f}, std={matrix.std():.6f}")
    print(f"  min={matrix.min():.6f}, max={matrix.max():.6f}")

    # Run SVD analysis
    ranks = [32, 64, 128, 256, 384, 512, 768, 1024]
    S = analyze_svd(matrix, tensor_name, ranks)

    # Quick analysis of singular value decay
    print(f"\n--- Singular Value Decay ---")
    energy = np.cumsum(S**2) / np.sum(S**2)
    for pct in [0.90, 0.95, 0.99, 0.999]:
        rank_needed = np.searchsorted(energy, pct) + 1
        print(f"  {pct*100:.1f}% energy captured at rank {rank_needed}/{len(S)}")


if __name__ == "__main__":
    main()
