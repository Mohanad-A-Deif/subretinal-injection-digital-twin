#!/usr/bin/env bash
# Per-episode CSV exports, McNemar contingency tables and figures
set -e
python3 -m sri_twin.export_csv
python3 -m sri_twin.figures tradeoff ablation regimes ood compute estimation
