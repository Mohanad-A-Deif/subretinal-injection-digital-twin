"""Closed-loop evaluation of the distilled policy, the safety filter and the
surrogate/hybrid planners on the test partition.
Usage: python3 -m sri_twin.run_comp policy|sur"""
import sys
from .run_exp import run, BASE


def main(which):
    from .surrogate import BlockSurrogate
    from .policy import DistilledPolicy
    sur = BlockSurrogate.load("results/models/surrogate_1.0.pt")
    pol = DistilledPolicy.load("results/models/policy.pt")
    if which == "policy":
        run("policy", dict(method="policy"), 8, 50, policy=pol)
        run("policy_safety", dict(method="policy", safety=True), 8, 50, policy=pol)
        run("joint_safety", dict(BASE, safety=True), 4, 50)
    else:
        run("sur_only", dict(BASE, inner="surrogate"), 4, 50, surrogate=sur)
        run("hybrid_1_4", dict(BASE, inner="hybrid", refine_every=4), 4, 50, surrogate=sur)
        run("hybrid_1_2", dict(BASE, inner="hybrid", refine_every=2), 4, 50, surrogate=sur)


if __name__ == "__main__":
    main(sys.argv[1])
