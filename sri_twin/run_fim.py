"""Fisher-information identifiability along closed-loop trajectories of the joint controller (test batch 50000)."""
import pickle, numpy as np
from .fim import fisher, summarize, NAMES


def main():
    Jac, Rinv, t, m = fisher(50_000, 50, dict(method="mpc", mode="joint", lam=0.75))
    res = {}
    for lab, tu in [("approach_1.5s", 1.5), ("full", None)]:
        s = summarize(Jac, Rinv, t, tu)
        res[lab] = {k: v for k, v in s.items() if k != "F"}
        res[lab]["F"] = s["F"]
        print(lab, "median cond", np.median(s["cond"]), "median CRLB", np.median(s["crlb"], 0))
    pickle.dump(res, open("results/logs/fim.pkl", "wb"))


if __name__ == "__main__":
    main()
