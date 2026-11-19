#!/usr/bin/env bash
# Full reproduction (about 16 min on a 2-core CPU). Run from the code/ directory.
set -e
cd "$(dirname "$0")"
L=../results/loss_tables
python precompute.py 1.0 $L/tgt_0.npz 0 2 & python precompute.py 1.0 $L/tgt_1.npz 1 2 & wait
python precompute.py 0.0 $L/uni_0.npz 0 2 & python precompute.py 0.0 $L/uni_1.npz 1 2 & wait
python run_experiments.py
python breakdown.py
python fig_framework.py
python fig_example.py
python fig_results.py
echo "Done: results/ and figures/ regenerated."
