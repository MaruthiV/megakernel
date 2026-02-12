"""
Python Bridge for Megakernel

Loads the compiled megakernel.so shared library via ctypes and provides
a simple Python interface for text generation with throughput measurement.

Usage:
    from host.bridge import MegakernelEngine

    engine = MegakernelEngine(
        weights_path="weights.bin",
        offsets_path="weights_offsets.json",
        lib_path="megakernel.so",
        tokenizer=tokenizer,  # HuggingFace tokenizer
    )

    text, tok_per_sec = engine.generate("The meaning of life is", max_tokens=100)
    print(text)
    print(f"{tok_per_sec:.1f} tok/s")
"""

import ctypes
import json
import time
import numpy as np
import torch


class MegakernelEngine:
    """Python wrapper around the megakernel shared library."""

    def __init__(self, weights_path, offsets_path, lib_path, tokenizer):
        self.tokenizer = tokenizer
        self.lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()
        self._load_weights(weights_path, offsets_path)
        self._init_kernel()

    def _setup_ctypes(self):
        """Declare C function signatures."""
        # megakernel_init(void* d_weights, long long* offsets, int num_offsets) -> void*
        self.lib.megakernel_init.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
        ]
        self.lib.megakernel_init.restype = ctypes.c_void_p

        # megakernel_decode(void* handle, int input_token) -> int
        self.lib.megakernel_decode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.megakernel_decode.restype = ctypes.c_int

        # megakernel_reset(void* handle)
        self.lib.megakernel_reset.argtypes = [ctypes.c_void_p]
        self.lib.megakernel_reset.restype = None

        # megakernel_get_pos(void* handle) -> int
        self.lib.megakernel_get_pos.argtypes = [ctypes.c_void_p]
        self.lib.megakernel_get_pos.restype = ctypes.c_int

        # megakernel_free(void* handle)
        self.lib.megakernel_free.argtypes = [ctypes.c_void_p]
        self.lib.megakernel_free.restype = None

        # megakernel_sync()
        self.lib.megakernel_sync.argtypes = []
        self.lib.megakernel_sync.restype = None

    def _load_weights(self, weights_path, offsets_path):
        """Load flat binary weights onto GPU and prepare offset array."""
        with open(offsets_path) as f:
            offsets_dict = json.load(f)

        with open(weights_path, "rb") as f:
            data = f.read()

        total_bytes = len(data)
        print(f"Loading {total_bytes / 1e6:.1f} MB of weights to GPU...")

        # Copy to GPU via PyTorch
        self._gpu_buffer = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
        cpu_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self._gpu_buffer.copy_(cpu_tensor)
        self._d_weights_ptr = self._gpu_buffer.data_ptr()

        # Build ordered offsets array matching the C API expectation:
        # [embedding, final_norm, layer.0.attn_norm, layer.0.w_q, ..., layer.27.w_down]
        layer_keys = ["attn_norm", "w_q", "w_k", "w_v", "w_o", "ffn_norm", "w_gate", "w_up", "w_down"]
        offsets_list = [offsets_dict["embedding"], offsets_dict["final_norm"]]
        for i in range(28):
            for key in layer_keys:
                offsets_list.append(offsets_dict[f"layer.{i}.{key}"])

        self._num_offsets = len(offsets_list)
        self._offsets_arr = (ctypes.c_longlong * self._num_offsets)(*offsets_list)
        print(f"Weight offsets prepared: {self._num_offsets} entries")

    def _init_kernel(self):
        """Initialize the CUDA kernel state."""
        self._handle = self.lib.megakernel_init(
            ctypes.c_void_p(self._d_weights_ptr),
            self._offsets_arr,
            self._num_offsets,
        )
        if not self._handle:
            raise RuntimeError("megakernel_init returned NULL")
        print("Kernel initialized successfully.")

    def reset(self):
        """Reset KV cache and barriers for a new generation."""
        self.lib.megakernel_reset(self._handle)

    def decode_token(self, token_id):
        """Run one forward pass: input token -> next token."""
        return self.lib.megakernel_decode(self._handle, token_id)

    def generate(self, prompt, max_tokens=100, print_stream=False):
        """
        Generate text from a prompt.

        Args:
            prompt: Input text string
            max_tokens: Maximum number of tokens to generate
            print_stream: If True, print tokens as they're generated

        Returns:
            (generated_text, tokens_per_second)
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)

        # Reset state for fresh generation
        self.reset()

        # Process prompt tokens (prefill — feed each token to build KV cache)
        for tok in input_ids:
            _ = self.decode_token(tok)

        self.lib.megakernel_sync()

        # Generate new tokens
        generated = []
        eos_id = self.tokenizer.eos_token_id

        start = time.perf_counter()

        last_token = input_ids[-1]
        # The last prefill step already produced a next token, but we fed all
        # prompt tokens through decode_token which advances position.
        # Now we're at position len(input_ids), ready to generate.

        # Actually, decode_token returns the next token prediction.
        # The last call in the prefill loop returned the first generated token.
        # Let's redo: reset and feed prompt, collecting the last output.
        self.reset()

        for i, tok in enumerate(input_ids[:-1]):
            self.decode_token(tok)

        # Feed last prompt token and get first generated token
        next_token = self.decode_token(input_ids[-1])
        self.lib.megakernel_sync()

        start = time.perf_counter()

        for i in range(max_tokens):
            generated.append(next_token)

            if print_stream:
                word = self.tokenizer.decode([next_token])
                print(word, end="", flush=True)

            if next_token == eos_id:
                break

            next_token = self.decode_token(next_token)

        self.lib.megakernel_sync()
        elapsed = time.perf_counter() - start

        if print_stream:
            print()

        # Compute throughput
        num_generated = len(generated)
        tok_per_sec = num_generated / elapsed if elapsed > 0 else 0

        # Decode full output
        full_text = self.tokenizer.decode(input_ids + generated, skip_special_tokens=True)

        return full_text, tok_per_sec

    def benchmark(self, num_tokens=50, prompt_token=1):
        """
        Benchmark raw decode throughput without tokenization overhead.

        Returns tokens_per_second.
        """
        # Warmup
        self.reset()
        token = prompt_token
        for _ in range(5):
            token = self.decode_token(token)
        self.lib.megakernel_sync()

        # Benchmark
        self.reset()
        self.lib.megakernel_sync()

        token = prompt_token
        start = time.perf_counter()
        for _ in range(num_tokens):
            token = self.decode_token(token)
        self.lib.megakernel_sync()
        elapsed = time.perf_counter() - start

        return num_tokens / elapsed

    def benchmark_sweep(self, positions=None):
        """
        Benchmark at multiple context positions.

        Returns list of dicts with position, tok_per_sec, latency_us, bw_util.
        """
        if positions is None:
            positions = [1, 10, 50, 100, 500, 1000, 2000]

        model_bytes = 926e6  # approximate BF16 model size
        # Read peak BW from kernel
        results = []

        for pos in positions:
            self.reset()
            token = 1

            # Fill KV cache to target position
            for _ in range(pos):
                token = self.decode_token(token)
            self.lib.megakernel_sync()

            # Benchmark a few tokens at this position
            bench_tokens = 20
            start = time.perf_counter()
            for _ in range(bench_tokens):
                token = self.decode_token(token)
            self.lib.megakernel_sync()
            elapsed = time.perf_counter() - start

            tps = bench_tokens / elapsed
            latency_us = elapsed * 1e6 / bench_tokens
            bw_gbps = model_bytes * tps / 1e9

            results.append({
                "position": pos,
                "tok_per_sec": tps,
                "latency_us": latency_us,
                "bw_gbps": bw_gbps,
            })

        return results

    def close(self):
        """Free GPU resources."""
        if self._handle:
            self.lib.megakernel_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
