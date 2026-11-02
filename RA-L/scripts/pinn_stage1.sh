#!/bin/bash
# PINN baseline, stage 1: forward and inverse PINNs on the 2-module SOFA data, three physics weights each
cd "$(dirname "$0")/.."
run() { .venv/bin/python -m tdcr.baselines.pinn --variant $1 --w-phys $2 --threads 5 --out results/pinn/2s_$1_w$2.json > results/pinn/2s_$1_w$2.log 2>&1; }
run forward 0.1 & run forward 1 & run forward 10 & wait
run inverse 0.1 & run inverse 1 & run inverse 10 & wait
echo stage1 done
