#pragma once
#include <stdint.h>

enum KvLayout : int {
  KV_LAYOUT_SEQ_HEAD_DIM = 0,   // [seq, n_heads, head_dim]
  KV_LAYOUT_HEAD_DIM_SEQ = 1,   // [n_heads, head_dim, seq]
};

struct KvLayoutConfig {
  KvLayout layout;
  int num_heads;
  int head_dim;
  int seq_len;
};

// TODO: Implement layout-specific index helpers and vectorized load helpers.
