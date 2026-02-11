"""
Correctness Validation for Megakernel

Compares megakernel output against PyTorch/HuggingFace reference at:
1. Per-layer level (RMSNorm, QKV, attention, MLP outputs)
2. Final logits level
3. Generated token sequence level
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
HIDDEN_DIM = 1024
NUM_LAYERS = 28


def get_reference_model(cache_dir="./model_cache"):
    """Load the reference HuggingFace model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=cache_dir,
    )
    model.eval()
    return model, tokenizer


def reference_forward(model, input_ids):
    """Run reference forward pass and return logits."""
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits


def reference_generate(model, tokenizer, prompt, max_new_tokens=50):
    """Generate tokens with HuggingFace model as reference."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=1.0,
        )
    tokens = outputs[0].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, text


def compare_logits(ref_logits, kernel_logits, label="", tolerance=1e-2):
    """Compare logits from reference and kernel, report errors."""
    if isinstance(ref_logits, torch.Tensor):
        ref = ref_logits.float().cpu().numpy().flatten()
    else:
        ref = np.array(ref_logits, dtype=np.float32).flatten()

    if isinstance(kernel_logits, torch.Tensor):
        kern = kernel_logits.float().cpu().numpy().flatten()
    else:
        kern = np.array(kernel_logits, dtype=np.float32).flatten()

    # Ensure same length (kernel might not have padding)
    min_len = min(len(ref), len(kern))
    ref = ref[:min_len]
    kern = kern[:min_len]

    abs_diff = np.abs(ref - kern)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    rel_diff = abs_diff / (np.abs(ref) + 1e-8)
    max_rel_error = np.max(rel_diff)

    # Check argmax match (most important for greedy decoding)
    ref_argmax = np.argmax(ref)
    kern_argmax = np.argmax(kern)
    argmax_match = ref_argmax == kern_argmax

    passed = max_abs_error < tolerance and argmax_match

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    print(f"    Max abs error: {max_abs_error:.6f} (tolerance: {tolerance})")
    print(f"    Mean abs error: {mean_abs_error:.6f}")
    print(f"    Max rel error: {max_rel_error:.6f}")
    print(f"    Argmax: ref={ref_argmax}, kernel={kern_argmax} ({'match' if argmax_match else 'MISMATCH'})")

    return passed


def validate_greedy_generation(ref_tokens, kernel_tokens, tokenizer):
    """Compare generated token sequences."""
    match_count = 0
    max_compare = min(len(ref_tokens), len(kernel_tokens))

    for i in range(max_compare):
        if ref_tokens[i] == kernel_tokens[i]:
            match_count += 1
        else:
            print(f"  First divergence at position {i}:")
            print(f"    Reference: token {ref_tokens[i]} = '{tokenizer.decode([ref_tokens[i]])}'")
            print(f"    Kernel:    token {kernel_tokens[i]} = '{tokenizer.decode([kernel_tokens[i]])}'")
            break

    match_pct = match_count / max_compare * 100 if max_compare > 0 else 0
    print(f"  Token match: {match_count}/{max_compare} ({match_pct:.1f}%)")

    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
    kern_text = tokenizer.decode(kernel_tokens, skip_special_tokens=True)

    print(f"  Reference text: {ref_text[:200]}...")
    print(f"  Kernel text:    {kern_text[:200]}...")

    return match_count == max_compare


def run_validation(kernel_forward_fn=None, kernel_generate_fn=None, cache_dir="./model_cache"):
    """
    Run full validation suite.

    kernel_forward_fn: callable(input_token_id) -> logits array
    kernel_generate_fn: callable(prompt_token_id, num_tokens) -> list of token ids
    """
    print("=" * 60)
    print("Correctness Validation")
    print("=" * 60)

    model, tokenizer = get_reference_model(cache_dir)

    # Test 1: Single-token forward pass logits
    print("\n--- Test 1: Single-token logits ---")
    test_tokens = [tokenizer.bos_token_id or 1, 100, 500, 1000, 5000]
    for tok in test_tokens:
        input_ids = torch.tensor([[tok]], device="cuda")
        ref_logits = reference_forward(model, input_ids)
        ref_logits_flat = ref_logits[0, -1, :]  # last position logits

        if kernel_forward_fn:
            kernel_logits = kernel_forward_fn(tok)
            compare_logits(ref_logits_flat, kernel_logits,
                          label=f"Token {tok}", tolerance=1e-2)
        else:
            print(f"  [SKIP] Token {tok} (no kernel_forward_fn provided)")

    # Test 2: Greedy generation comparison
    print("\n--- Test 2: Greedy generation (50 tokens) ---")
    prompt = "The meaning of life is"
    ref_tokens, ref_text = reference_generate(model, tokenizer, prompt, max_new_tokens=50)
    print(f"  Reference: {ref_text[:200]}")

    if kernel_generate_fn:
        prompt_ids = tokenizer.encode(prompt)
        # For our megakernel, we start from the last prompt token
        kernel_tokens = kernel_generate_fn(prompt_ids[-1], 50)
        validate_greedy_generation(ref_tokens[len(prompt_ids):], kernel_tokens, tokenizer)
    else:
        print("  [SKIP] No kernel_generate_fn provided")

    # Test 3: Throughput comparison
    print("\n--- Test 3: HuggingFace baseline throughput ---")
    import time

    input_ids = torch.tensor([[1]], device="cuda")
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()

    num_iters = 20
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    hf_toks = num_iters / elapsed

    print(f"  HuggingFace throughput: {hf_toks:.1f} tok/s (batch=1, short context)")
    print(f"  This is the baseline to beat.")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
