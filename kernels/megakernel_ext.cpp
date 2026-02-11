#include <torch/extension.h>
#include <vector>

void megakernel_forward_cuda(
    torch::Tensor input,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor output,
    int64_t num_layers,
    int64_t position,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta);

void megakernel_forward_seq_cuda(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor outputs,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block);

void megakernel_forward_seq_cuda_bf16(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor outputs,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block);

void megakernel_forward_seq_cuda_int8kv(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor outputs,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block);

torch::Tensor megakernel_forward_seq_int8kv(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block) {
  if (!inputs.is_cuda()) {
    throw std::invalid_argument("megakernel_forward_seq_int8kv expects CUDA tensors");
  }
  if (inputs.scalar_type() != torch::kFloat32) {
    inputs = inputs.to(torch::kFloat32);
  }
  if (!inputs.is_contiguous()) {
    inputs = inputs.contiguous();
  }
  auto outputs = torch::zeros_like(inputs);
  megakernel_forward_seq_cuda_int8kv(
      inputs,
      rms_attn_weight,
      rms_ffn_weight,
      w_qkv,
      w_o,
      w_gate,
      w_up,
      w_down,
      kv_k,
      kv_v,
      k_scale,
      v_scale,
      outputs,
      num_layers,
      start_pos,
      num_heads,
      num_kv_heads,
      head_dim,
      kv_layout,
      eps,
      rope_theta,
      threads_per_block);
  return outputs;
}
void megakernel_forward_seq_persistent_cuda(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor outputs,
    torch::Tensor q_buf,
    torch::Tensor attn_buf,
    torch::Tensor barrier_counter,
    torch::Tensor barrier_sense,
    torch::Tensor sync_flags,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t num_blocks,
    int64_t threads_per_block);

void megakernel_forward_seq_persistent_cuda_int8kv(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor outputs,
    torch::Tensor q_buf,
    torch::Tensor attn_buf,
    torch::Tensor barrier_counter,
    torch::Tensor barrier_sense,
    torch::Tensor sync_flags,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t num_blocks,
    int64_t threads_per_block);
torch::Tensor megakernel_forward(
    torch::Tensor input,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    int64_t num_layers,
    int64_t position,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block) {
  if (!input.is_cuda()) {
    throw std::invalid_argument("megakernel_forward expects CUDA tensors");
  }
  if (input.scalar_type() != torch::kFloat32) {
    input = input.to(torch::kFloat32);
  }
  if (rms_attn_weight.scalar_type() != torch::kFloat32) {
    rms_attn_weight = rms_attn_weight.to(torch::kFloat32);
  }
  if (rms_ffn_weight.scalar_type() != torch::kFloat32) {
    rms_ffn_weight = rms_ffn_weight.to(torch::kFloat32);
  }
  if (w_qkv.scalar_type() != torch::kFloat32) {
    w_qkv = w_qkv.to(torch::kFloat32);
  }
  if (w_o.scalar_type() != torch::kFloat32) {
    w_o = w_o.to(torch::kFloat32);
  }
  if (w_gate.scalar_type() != torch::kFloat32) {
    w_gate = w_gate.to(torch::kFloat32);
  }
  if (w_up.scalar_type() != torch::kFloat32) {
    w_up = w_up.to(torch::kFloat32);
  }
  if (w_down.scalar_type() != torch::kFloat32) {
    w_down = w_down.to(torch::kFloat32);
  }
  if (kv_k.scalar_type() != torch::kFloat32) {
    kv_k = kv_k.to(torch::kFloat32);
  }
  if (kv_v.scalar_type() != torch::kFloat32) {
    kv_v = kv_v.to(torch::kFloat32);
  }
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }
  if (!rms_attn_weight.is_contiguous()) {
    rms_attn_weight = rms_attn_weight.contiguous();
  }
  if (!rms_ffn_weight.is_contiguous()) {
    rms_ffn_weight = rms_ffn_weight.contiguous();
  }
  if (!w_qkv.is_contiguous()) {
    w_qkv = w_qkv.contiguous();
  }
  if (!w_o.is_contiguous()) {
    w_o = w_o.contiguous();
  }
  if (!w_up.is_contiguous()) {
    w_up = w_up.contiguous();
  }
  if (!w_gate.is_contiguous()) {
    w_gate = w_gate.contiguous();
  }
  if (!w_down.is_contiguous()) {
    w_down = w_down.contiguous();
  }
  if (!kv_k.is_contiguous()) {
    kv_k = kv_k.contiguous();
  }
  if (!kv_v.is_contiguous()) {
    kv_v = kv_v.contiguous();
  }
  auto output = torch::zeros_like(input);
  megakernel_forward_cuda(
      input,
      rms_attn_weight,
      rms_ffn_weight,
      w_qkv,
      w_o,
      w_gate,
      w_up,
      w_down,
      kv_k,
      kv_v,
      output,
      num_layers,
      position,
      num_heads,
      num_kv_heads,
      head_dim,
      kv_layout,
      eps,
      rope_theta);
  return output;
}

torch::Tensor megakernel_forward_seq(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta) {
  if (!inputs.is_cuda()) {
    throw std::invalid_argument("megakernel_forward_seq expects CUDA tensors");
  }
  if (inputs.scalar_type() != torch::kFloat32 && inputs.scalar_type() != torch::kBFloat16) {
    inputs = inputs.to(torch::kFloat32);
  }
  if (!inputs.is_contiguous()) {
    inputs = inputs.contiguous();
  }
  auto outputs = torch::zeros_like(inputs);
  if (inputs.scalar_type() == torch::kBFloat16) {
    megakernel_forward_seq_cuda_bf16(
        inputs,
        rms_attn_weight,
        rms_ffn_weight,
        w_qkv,
        w_o,
        w_gate,
        w_up,
        w_down,
        kv_k,
        kv_v,
        outputs,
        num_layers,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_layout,
        eps,
        rope_theta,
        threads_per_block);
  } else {
    megakernel_forward_seq_cuda(
        inputs,
        rms_attn_weight,
        rms_ffn_weight,
        w_qkv,
        w_o,
        w_gate,
        w_up,
        w_down,
        kv_k,
        kv_v,
        outputs,
        num_layers,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_layout,
        eps,
        rope_theta,
        threads_per_block);
  }
  return outputs;
}

torch::Tensor megakernel_forward_seq_persistent(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor q_buf,
    torch::Tensor attn_buf,
    torch::Tensor barrier_counter,
    torch::Tensor barrier_sense,
    torch::Tensor sync_flags,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t num_blocks,
    int64_t threads_per_block) {
  if (!inputs.is_cuda()) {
    throw std::invalid_argument("megakernel_forward_seq_persistent expects CUDA tensors");
  }
  if (inputs.scalar_type() != torch::kFloat32) {
    inputs = inputs.to(torch::kFloat32);
  }
  if (!inputs.is_contiguous()) {
    inputs = inputs.contiguous();
  }
  auto outputs = torch::zeros_like(inputs);
  megakernel_forward_seq_persistent_cuda(
      inputs,
      rms_attn_weight,
      rms_ffn_weight,
      w_qkv,
      w_o,
      w_gate,
      w_up,
      w_down,
      kv_k,
      kv_v,
      outputs,
      q_buf,
      attn_buf,
      barrier_counter,
      barrier_sense,
      sync_flags,
      num_layers,
      start_pos,
      num_heads,
      num_kv_heads,
      head_dim,
      kv_layout,
      eps,
      rope_theta,
      num_blocks,
      threads_per_block);
  return outputs;
}

torch::Tensor megakernel_forward_seq_persistent_int8kv(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor q_buf,
    torch::Tensor attn_buf,
    torch::Tensor barrier_counter,
    torch::Tensor barrier_sense,
    torch::Tensor sync_flags,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t num_blocks,
    int64_t threads_per_block) {
  if (!inputs.is_cuda()) {
    throw std::invalid_argument("megakernel_forward_seq_persistent_int8kv expects CUDA tensors");
  }
  if (inputs.scalar_type() != torch::kFloat32) {
    inputs = inputs.to(torch::kFloat32);
  }
  if (!inputs.is_contiguous()) {
    inputs = inputs.contiguous();
  }
  auto outputs = torch::zeros_like(inputs);
  megakernel_forward_seq_persistent_cuda_int8kv(
      inputs,
      rms_attn_weight,
      rms_ffn_weight,
      w_qkv,
      w_o,
      w_gate,
      w_up,
      w_down,
      kv_k,
      kv_v,
      k_scale,
      v_scale,
      outputs,
      q_buf,
      attn_buf,
      barrier_counter,
      barrier_sense,
      sync_flags,
      num_layers,
      start_pos,
      num_heads,
      num_kv_heads,
      head_dim,
      kv_layout,
      eps,
      rope_theta,
      num_blocks,
      threads_per_block);
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("megakernel_forward", &megakernel_forward, "Megakernel forward (CUDA)");
  m.def("megakernel_forward_seq", &megakernel_forward_seq, "Megakernel forward seq (CUDA)");
  m.def("megakernel_forward_seq_int8kv", &megakernel_forward_seq_int8kv,
        "Megakernel forward seq int8 kv (CUDA)");
  m.def("megakernel_forward_seq_persistent", &megakernel_forward_seq_persistent,
        "Megakernel forward seq persistent (CUDA)");
  m.def("megakernel_forward_seq_persistent_int8kv", &megakernel_forward_seq_persistent_int8kv,
        "Megakernel forward seq persistent int8 kv (CUDA)");
}
