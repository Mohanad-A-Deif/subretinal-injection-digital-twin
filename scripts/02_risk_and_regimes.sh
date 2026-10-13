#!/usr/bin/env bash
# Risk-weight sweep (200 episodes) and uncertainty regimes (200 episodes per regime)
set -e
python3 -m sri_twin.run_exp lam0.25 lam0.50 lam1.00 lam1.50 --nb 4
python3 -m sri_twin.run_exp nominal traj inf joint --nb 4 --regime low
python3 -m sri_twin.run_exp nominal traj inf joint lam0.00 --nb 4 --regime high
