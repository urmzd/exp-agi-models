# TODO

## Training runs needed
- [ ] `v11b_hard_routing` — hard Gumbel routing, multi-timescale linattn, adaptive halting
- [ ] `v11a_mixed_ops` — five fixed primitive ops composed sequentially
- [ ] `v12_vocab_slice` — processing in fixed k-length vocab-id slices
- [ ] `v10_state_cond_op` — state-conditioned soft read/op/write dispatch
- [ ] `v7_soft_ops` — differentiable register machine (address instability)
- [ ] `v8_lowrank_vv` — extended run at rank 8, measure long-range behavior

## Infrastructure
- [ ] MLX support for current models — only v0 has an MLX training script
- [ ] Wandb/tensorboard logging
- [ ] Add `v11b_hard_routing`, `v11a_mixed_ops`, `v12_vocab_slice` to `run_all.py` model list

## Training
- [x] Checkpoint save/resume
- [x] Roundtrip eval optional (ROUNDTRIP_EVAL=1)
- [ ] Learning rate warmup schedule (currently flat after warmup steps)
- [ ] Gumbel temperature annealing for `v11b_hard_routing` — anneal tau from 1.0 → 0.1 during training
