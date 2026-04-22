#!/usr/bin/env python3
"""Run all model versions sequentially and collect results."""
import os
import subprocess
import sys


MODELS = [
    {"version": "v1_shared_attn",     "env": {"NUM_STEPS": "8"}},
    {"version": "v2_conv",            "env": {"NUM_STEPS": "16"}},
    {"version": "v3_fourier_linattn", "env": {}},
    {"version": "v4_weight_shared",   "env": {}},
    {"version": "v5_fft_linattn",     "env": {"N_FOURIER_BASIS": "64"}},
    {"version": "v6_banded_fourier",  "env": {}},
    {"version": "v7_soft_ops",        "env": {"NUM_STEPS": "16", "N_CHANNELS": "64"}},
    {"version": "v8_lowrank_vv",      "env": {}},
    {"version": "v9_linattn",         "env": {}},
    {"version": "v10_state_cond_op",  "env": {}},
    {"version": "v11a_mixed_ops",     "env": {}},
    {"version": "v11b_hard_routing",  "env": {}},
    {"version": "v12_vocab_slice",    "env": {}},
    {"version": "v14_data_dependent", "env": {}},
    {"version": "v15_aux_loss",       "env": {}},
    {"version": "v16_multi_branch",   "env": {}},
]

# Detect GPU count
try:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    n_gpus = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
except FileNotFoundError:
    n_gpus = 1


def main():
    batch = os.environ.get("TRAIN_BATCH_TOKENS", "491520")
    grad_accum = os.environ.get("GRAD_ACCUM_STEPS", "16")
    log_every = os.environ.get("TRAIN_LOG_EVERY", "50")
    iterations = os.environ.get("ITERATIONS", "500")

    results = []
    for m in MODELS:
        version = m["version"]
        run_id = f"{version}_eval"
        print(f"\n{'='*60}")
        print(f"  Running {version} (run_id={run_id})")
        print(f"{'='*60}\n")

        env = {
            **os.environ,
            "MODEL_VERSION": version,
            "TRAIN_BATCH_TOKENS": batch,
            "GRAD_ACCUM_STEPS": grad_accum,
            "TRAIN_LOG_EVERY": log_every,
            "ITERATIONS": iterations,
            "RUN_ID": run_id,
            **m["env"],
        }

        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={n_gpus}",
            "train.py",
        ]

        ret = subprocess.run(cmd, env=env)
        status = "OK" if ret.returncode == 0 else f"FAIL({ret.returncode})"
        results.append((version, status))
        print(f"\n  {version}: {status}\n")

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for version, status in results:
        print(f"  {version:10s} {status}")

    # Print results table
    print(f"\n{'='*60}")
    print("  Results Table")
    print(f"{'='*60}\n")
    subprocess.run([sys.executable, "results.py"])


if __name__ == "__main__":
    main()
