#!/bin/bash
# PINN baseline, stage 2: 4-module physics-only retraining for the validation-selected run of each variant
cd "$(dirname "$0")/.."
.venv/bin/python -m tdcr.baselines.pinn --four-from results/pinn/2s_forward_w0.1.json --threads 7 --out results/pinn/4s_forward.json > results/pinn/4s_forward.log 2>&1 &
.venv/bin/python -m tdcr.baselines.pinn --four-from results/pinn/2s_inverse_w10.json --threads 7 --out results/pinn/4s_inverse.json > results/pinn/4s_inverse.log 2>&1 &
wait
echo stage2 done
