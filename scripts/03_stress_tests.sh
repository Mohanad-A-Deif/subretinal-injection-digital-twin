#!/usr/bin/env bash
# Structural mismatch, delay and dropout tests (150 episodes each) and sensing perturbations
set -e
for v in puncture compliance saccade heavydelay driftdelay dropout shift; do
  python3 -m sri_twin.run_exp nominal inf joint --nb 3 --variant $v
done
python3 -m sri_twin.run_exp lam0.00 --nb 3 --variant puncture
python3 -m sri_twin.run_exp abl_adaptive --nb 3 --variant shift
for s in noise0.5 noise2 noise4 biasF biasD; do
  python3 -m sri_twin.run_exp joint nominal --nb 3 --sensing $s
done
