# Profiling Notes (Nsight Systems / Nsight Compute)

Recommended flow:
1. Use Nsight Systems to identify kernel time, sync, and launch overhead.
2. Use Nsight Compute to inspect memory throughput, occupancy, and instruction mix.

Commands (examples):
- Nsight Systems:
  nsys profile --trace=cuda,nvtx -o profile_nsys python bench/run.py --gpu H100 --pos 1,4096 --tokens 16

- Nsight Compute (single kernel):
  ncu --set full -o profile_ncu python -c "from megakernel import run_decode_steps; import torch; from megakernel.launch import load_config; cfg=load_config('configs/qwen3_0p6b_bf16.yaml'); x=torch.randn(16,cfg['model']['hidden_size'],device='cuda'); run_decode_steps(cfg,None,None,x,0)"
