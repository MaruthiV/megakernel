"""
Weight Loader for Qwen3-0.6B Megakernel

Downloads Qwen3-0.6B from HuggingFace and converts weights to a flat binary
format that the CUDA kernel can mmap directly.

Weight layout (all BF16):
  - embedding: [VOCAB_SIZE, HIDDEN_DIM] = [151936, 1024]
  - final_norm: [HIDDEN_DIM] = [1024]
  - For each layer 0..27:
    - attn_norm:  [HIDDEN_DIM]
    - w_q:        [Q_DIM, HIDDEN_DIM]       = [2048, 1024]  (transposed for GEMV)
    - w_k:        [KV_DIM, HIDDEN_DIM]      = [1024, 1024]
    - w_v:        [KV_DIM, HIDDEN_DIM]      = [1024, 1024]
    - q_norm:     [HEAD_DIM]                = [128]  (QK-norm, Qwen3-specific)
    - k_norm:     [HEAD_DIM]                = [128]  (QK-norm, Qwen3-specific)
    - w_o:        [HIDDEN_DIM, Q_DIM]       = [1024, 2048]
    - ffn_norm:   [HIDDEN_DIM]
    - w_gate:     [INTERMEDIATE_DIM, HIDDEN_DIM] = [3072, 1024]
    - w_up:       [INTERMEDIATE_DIM, HIDDEN_DIM] = [3072, 1024]
    - w_down:     [HIDDEN_DIM, INTERMEDIATE_DIM] = [1024, 3072]

Note: Weight matrices are stored as [output_dim, input_dim] (row-major).
For GEMV (output = weight @ input), each row of the weight matrix corresponds
to one output element. The CUDA kernel reads row-by-row with coalesced access
along the input dimension.
"""

import os
import struct
import numpy as np
import torch

# Model constants (must match config.cuh)
HIDDEN_DIM = 1024
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_DIM = 3072
VOCAB_SIZE = 151936
Q_DIM = NUM_Q_HEADS * HEAD_DIM     # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM   # 1024

MODEL_ID = "Qwen/Qwen3-0.6B"


def download_model(cache_dir="./model_cache"):
    """Download Qwen3-0.6B from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    print("Download complete.")
    return model, tokenizer


def extract_weights(model):
    """Extract weights from PyTorch model into a dictionary of numpy arrays."""
    state = model.state_dict()
    weights = {}

    # Embedding (tied with lm_head)
    weights["embedding"] = state["model.embed_tokens.weight"].to(torch.bfloat16).cpu()

    # Final norm
    weights["final_norm"] = state["model.norm.weight"].to(torch.bfloat16).cpu()

    # Per-layer weights
    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}"

        # Attention norm
        weights[f"layer.{i}.attn_norm"] = state[f"{prefix}.input_layernorm.weight"].to(torch.bfloat16).cpu()

        # QKV projections
        # Qwen3 stores Q, K, V as separate projections
        w_q = state[f"{prefix}.self_attn.q_proj.weight"].to(torch.bfloat16).cpu()
        w_k = state[f"{prefix}.self_attn.k_proj.weight"].to(torch.bfloat16).cpu()
        w_v = state[f"{prefix}.self_attn.v_proj.weight"].to(torch.bfloat16).cpu()

        # Qwen3 may have QK norm (q_norm, k_norm)
        # Check if they exist
        q_norm_key = f"{prefix}.self_attn.q_norm.weight"
        k_norm_key = f"{prefix}.self_attn.k_norm.weight"
        if q_norm_key in state:
            weights[f"layer.{i}.q_norm"] = state[q_norm_key].to(torch.bfloat16).cpu()
            weights[f"layer.{i}.k_norm"] = state[k_norm_key].to(torch.bfloat16).cpu()

        weights[f"layer.{i}.w_q"] = w_q  # [Q_DIM, HIDDEN_DIM]
        weights[f"layer.{i}.w_k"] = w_k  # [KV_DIM, HIDDEN_DIM]
        weights[f"layer.{i}.w_v"] = w_v  # [KV_DIM, HIDDEN_DIM]

        # Output projection
        weights[f"layer.{i}.w_o"] = state[f"{prefix}.self_attn.o_proj.weight"].to(torch.bfloat16).cpu()

        # FFN norm
        weights[f"layer.{i}.ffn_norm"] = state[f"{prefix}.post_attention_layernorm.weight"].to(torch.bfloat16).cpu()

        # MLP weights (SwiGLU: gate, up, down)
        weights[f"layer.{i}.w_gate"] = state[f"{prefix}.mlp.gate_proj.weight"].to(torch.bfloat16).cpu()
        weights[f"layer.{i}.w_up"] = state[f"{prefix}.mlp.up_proj.weight"].to(torch.bfloat16).cpu()
        weights[f"layer.{i}.w_down"] = state[f"{prefix}.mlp.down_proj.weight"].to(torch.bfloat16).cpu()

    return weights


def validate_shapes(weights):
    """Verify all weight shapes match expected model config."""
    assert weights["embedding"].shape == (VOCAB_SIZE, HIDDEN_DIM), \
        f"embedding shape mismatch: {weights['embedding'].shape}"
    assert weights["final_norm"].shape == (HIDDEN_DIM,), \
        f"final_norm shape mismatch: {weights['final_norm'].shape}"

    for i in range(NUM_LAYERS):
        shapes = {
            f"layer.{i}.attn_norm": (HIDDEN_DIM,),
            f"layer.{i}.w_q": (Q_DIM, HIDDEN_DIM),
            f"layer.{i}.w_k": (KV_DIM, HIDDEN_DIM),
            f"layer.{i}.w_v": (KV_DIM, HIDDEN_DIM),
            f"layer.{i}.q_norm": (HEAD_DIM,),
            f"layer.{i}.k_norm": (HEAD_DIM,),
            f"layer.{i}.w_o": (Q_DIM, HIDDEN_DIM),
            f"layer.{i}.ffn_norm": (HIDDEN_DIM,),
            f"layer.{i}.w_gate": (INTERMEDIATE_DIM, HIDDEN_DIM),
            f"layer.{i}.w_up": (INTERMEDIATE_DIM, HIDDEN_DIM),
            f"layer.{i}.w_down": (HIDDEN_DIM, INTERMEDIATE_DIM),
        }
        for key, expected_shape in shapes.items():
            actual = weights[key].shape
            assert actual == expected_shape, f"{key} shape mismatch: {actual} vs {expected_shape}"

    print("All weight shapes validated successfully!")


def save_flat_binary(weights, output_path="weights.bin"):
    """
    Save weights as a flat binary file in the order the CUDA kernel expects.

    Layout:
      1. embedding [VOCAB_SIZE, HIDDEN_DIM] bf16
      2. final_norm [HIDDEN_DIM] bf16
      3. For each layer:
         a. attn_norm [HIDDEN_DIM] bf16
         b. w_q [Q_DIM, HIDDEN_DIM] bf16
         c. w_k [KV_DIM, HIDDEN_DIM] bf16
         d. w_v [KV_DIM, HIDDEN_DIM] bf16
         e. q_norm [HEAD_DIM] bf16
         f. k_norm [HEAD_DIM] bf16
         g. w_o [Q_DIM, HIDDEN_DIM] bf16
         h. ffn_norm [HIDDEN_DIM] bf16
         i. w_gate [INTERMEDIATE_DIM, HIDDEN_DIM] bf16
         j. w_up [INTERMEDIATE_DIM, HIDDEN_DIM] bf16
         k. w_down [HIDDEN_DIM, INTERMEDIATE_DIM] bf16

    Also saves a metadata header with offsets for each weight.
    """
    parts = []
    offsets = {}
    current_offset = 0

    def add_weight(name, tensor):
        nonlocal current_offset
        data = tensor.contiguous().view(torch.uint16).numpy()  # bf16 as raw uint16 bytes
        offsets[name] = current_offset
        parts.append(data.tobytes())
        current_offset += len(parts[-1])

    # Embedding
    add_weight("embedding", weights["embedding"])

    # Final norm
    add_weight("final_norm", weights["final_norm"])

    # Per-layer weights (11 per layer, must match CUDA kernel's extern C init order)
    for i in range(NUM_LAYERS):
        add_weight(f"layer.{i}.attn_norm", weights[f"layer.{i}.attn_norm"])
        add_weight(f"layer.{i}.w_q", weights[f"layer.{i}.w_q"])
        add_weight(f"layer.{i}.w_k", weights[f"layer.{i}.w_k"])
        add_weight(f"layer.{i}.w_v", weights[f"layer.{i}.w_v"])
        add_weight(f"layer.{i}.q_norm", weights[f"layer.{i}.q_norm"])
        add_weight(f"layer.{i}.k_norm", weights[f"layer.{i}.k_norm"])
        add_weight(f"layer.{i}.w_o", weights[f"layer.{i}.w_o"])
        add_weight(f"layer.{i}.ffn_norm", weights[f"layer.{i}.ffn_norm"])
        add_weight(f"layer.{i}.w_gate", weights[f"layer.{i}.w_gate"])
        add_weight(f"layer.{i}.w_up", weights[f"layer.{i}.w_up"])
        add_weight(f"layer.{i}.w_down", weights[f"layer.{i}.w_down"])

    # Write binary file
    with open(output_path, "wb") as f:
        f.write(b"".join(parts))

    total_bytes = current_offset
    print(f"Saved weights to {output_path}: {total_bytes / 1e6:.1f} MB")

    # Save offsets as a separate JSON for the launcher
    import json
    offsets_path = output_path.replace(".bin", "_offsets.json")
    with open(offsets_path, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"Saved offsets to {offsets_path}")

    return offsets


def load_flat_binary_to_gpu(weights_path="weights.bin", offsets_path=None):
    """Load flat binary weights into GPU memory and return pointers."""
    import ctypes

    if offsets_path is None:
        offsets_path = weights_path.replace(".bin", "_offsets.json")

    import json
    with open(offsets_path) as f:
        offsets = json.load(f)

    # Read entire weight file
    with open(weights_path, "rb") as f:
        data = f.read()

    total_bytes = len(data)
    print(f"Loading {total_bytes / 1e6:.1f} MB of weights to GPU...")

    # Allocate GPU memory and copy
    import torch
    gpu_buffer = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    gpu_buffer.copy_(torch.frombuffer(data, dtype=torch.uint8))

    base_ptr = gpu_buffer.data_ptr()

    return gpu_buffer, base_ptr, offsets


def main():
    """Main: download model, extract weights, save binary."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights.bin", help="Output path for weight binary")
    parser.add_argument("--cache-dir", default="./model_cache", help="HF cache directory")
    args = parser.parse_args()

    model, tokenizer = download_model(args.cache_dir)
    weights = extract_weights(model)
    validate_shapes(weights)
    save_flat_binary(weights, args.output)

    print("\nDone! Weight binary ready for megakernel.")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model config: {model.config}")


if __name__ == "__main__":
    main()
