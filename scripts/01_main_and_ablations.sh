#!/usr/bin/env bash
# Main comparison and ablations (400 test episodes = 8 batches x 50)
set -e
python3 -m sri_twin.run_exp nominal traj inf joint --nb 8
python3 -m sri_twin.run_exp lam0.00 abl_nomap abl_decoupled abl_adaptive --nb 8
