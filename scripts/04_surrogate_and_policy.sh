#!/usr/bin/env bash
# Surrogate data generation, learning curve and planning fidelity; DAgger policy distillation;
# closed-loop evaluation of surrogate/hybrid planners, policy and safety filter
set -e
python3 -m sri_twin.collect train      # results/sur_train.npz (4.0e5 transitions, seeds 20000+)
python3 -m sri_twin.collect val        # results/sur_val.npz   (5.0e4 transitions, seeds 30000+)
python3 -m sri_twin.train_surrogate 8  # learning curve, results/surrogate_*.pt, plan_problems.pkl
python3 -m sri_twin.policy             # results/policy.pt (DAgger, seeds 40000+)
python3 -m sri_twin.run_comp policy    # policy, policy_safety, joint_safety
python3 -m sri_twin.run_comp sur       # sur_only, hybrid_1_4, hybrid_1_2
