#!/usr/bin/env bash
# v1.1.0: fixed-motion planners, missed planning deadlines, contact-robust estimators,
# weakly identifiable pair at the prior (all on the same held-out test seeds)
set -e
export PYTHONPATH=.
python3 -m sri_twin.run_exp joint_fixdepth joint_fixmotion joint_fixmotion_anchored --nb 8
python3 -m sri_twin.run_exp joint_fixmotion --nb 4 --regime high
python3 -m sri_twin.run_exp joint_fixmotion --nb 4 --regime low
python3 -m sri_twin.run_exp joint_miss1 joint_miss5 joint_missburst --nb 8
python3 -m sri_twin.run_exp joint_missburst --nb 3 --variant heavydelay
python3 -m sri_twin.run_exp joint_anchored joint_switched joint_fixweak --nb 8
python3 -m sri_twin.run_exp joint_anchored joint_anchored_norup joint_switched joint_fixmotion joint_fixmotion_anchored joint_fixmotion_switched --nb 3 --variant puncture
for v in compliance saccade heavydelay driftdelay dropout shift; do
  python3 -m sri_twin.run_exp joint_anchored joint_switched --nb 3 --variant $v
done
python3 -m sri_twin.run_exp joint_fixweak --nb 3 --variant shift
python3 -m sri_twin.export_csv
python3 -c "import sri_twin.figures as f; f.fig_tradeoff(); f.fig_ablation(); f.fig_regimes(); f.fig_ood()"
