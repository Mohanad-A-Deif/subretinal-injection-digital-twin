#!/usr/bin/env bash
# Estimation logs, Fisher identifiability and single-procedure latency (run on an idle machine)
set -e
python3 - <<'PY'
from sri_twin.run_exp import run, BASE
run('joint_est', dict(BASE), 2, 50, log_est=True)
run('joint_adapt_est', dict(BASE, est_mode='adaptive'), 2, 50, log_est=True, variant=dict(shift=True), tag='joint_shift_est')
run('joint_fixed_shift_est', dict(BASE), 2, 50, log_est=True, variant=dict(shift=True), tag='joint_fixed_shift_est')
PY
python3 -m sri_twin.run_fim
python3 -m sri_twin.latency
