# sri-twin: risk-aware joint control of needle motion and infusion for robotic subretinal injection (simulation)

This repository contains the complete simulation code, configurations, trained models and per-episode results for the manuscript

> *Uncertainty-Aware Digital Twin of Robotic Subretinal Injection for Optimal Needle Motion and Controlled Bleb Formation* (under review).

All results are generated in simulation. The digital twin is a reduced-order model whose parameters are either anchored to published measurements or declared as design assumptions; it has not been calibrated against bench, phantom or ex vivo data.

## Contents

| Path | Content |
|---|---|
| `sri_twin/params.py` | Nominal constants, uncertainty distributions, regimes, sensor noise, prior of the identified block |
| `sri_twin/core.py` | Twin equations (needle dynamics, Kelvin--Voigt contact, delivery zone, tract openness, reflux, leakage, bleb volume/area/pressure); structural variants used in the stress tests |
| `sri_twin/plant.py` | Truth plant at 1 ms: actuation delay, retinal motion (harmonics + Ornstein--Uhlenbeck), delayed noisy sensing, drift, dropout, delay and shift variants |
| `sri_twin/estimator.py` | Augmented-state UKF (13 states, 5 estimated parameters), delayed-measurement fusion, fixed / adaptive / frozen / gated modes |
| `sri_twin/controller.py` | Posterior scenario sampler, full-twin rollouts, risk score, CVaR, cross-entropy planner, execution law, nominal controller |
| `sri_twin/surrogate.py`, `collect.py`, `train_surrogate.py` | Block surrogate, training-data generation, learning curve and planning fidelity |
| `sri_twin/policy.py` | DAgger distillation of the planner into an explicit 20 ms policy |
| `sri_twin/episode.py` | Closed-loop runner (vectorized over episodes), safety filter |
| `sri_twin/run_exp.py`, `run_comp.py`, `run_fim.py`, `latency.py` | Experiment drivers |
| `sri_twin/fim.py` | Fisher-information identifiability |
| `sri_twin/analysis.py`, `export_csv.py`, `figures.py` | Statistics, CSV/contingency export, figures |
| `sri_twin/tune_exp.py` | Same driver on the tuning partition (used only for weight selection) |
| `scripts/` | Shell scripts reproducing every experiment in the order of the manuscript |
| `results/*.npz` | Per-episode metrics of every configuration (NumPy archives) |
| `results/csv/` | The same per-episode metrics as CSV, plus planner wall times |
| `results/mcnemar_contingency.json` | Paired 2x2 counts and Holm-adjusted tests for every reported comparison |
| `results/models/` | Trained surrogate (all learning-curve sizes) and distilled policy |
| `results/logs/` | Latency (raw per-decision times), Fisher analysis, surrogate learning curve, estimation logs, data-generation metadata |

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All reported runs used Python 3.13 on a 2-vCPU Intel Xeon (2.8 GHz), 7 GB RAM, CPU only.

## Reproducing the results

Run the scripts from the repository root, in order:

```bash
bash scripts/01_main_and_ablations.sh          # main comparison and ablations (400 episodes)
bash scripts/02_risk_and_regimes.sh            # risk-weight sweep, uncertainty regimes (200 episodes)
bash scripts/03_stress_tests.sh                # structural mismatch, delays, dropout, sensing (150 episodes)
bash scripts/04_surrogate_and_policy.sh        # surrogate data, training, fidelity; DAgger policy; closed loop
bash scripts/05_estimation_identifiability_latency.sh
bash scripts/06_tables_figures_exports.sh
```

`run_exp.py` skips any configuration whose result file already exists, so delete `results/<tag>.npz` to recompute it. Approximate cost on the reference machine: about 2.5 min of CPU time per batch of 50 planner episodes and a few seconds for the nominal policy.

### Seed partitions

| Partition | Seeds | Use |
|---|---|---|
| Tuning | 10 000 + batch | Weight selection only (`tune_exp.py`) |
| Surrogate training | 20 000 + batch | `collect.py train` |
| Surrogate validation | 30 000 + batch (and 31 000 for planning problems) | `collect.py val`, `train_surrogate.py` |
| Policy training | 40 000 + iteration x 100 + batch | `policy.py` |
| Test | 50 000 + batch | All reported results; latency uses 51 000-51 004 |

Each batch draws its parameters with `numpy.random.default_rng(seed)` and its process and sensor noise with `default_rng(seed + 7919)`, so all controllers see identical scenarios (common random numbers) and every comparison is paired.

### Result tags

`nominal`, `traj`, `inf`, `joint` are the four controllers; `lam*` the risk-weight sweep; `abl_nomap`, `abl_decoupled`, `abl_adaptive` the ablations; `*_low`, `*_high` the uncertainty regimes; `*_ood_<variant>` the structural and delay tests; `*_sens_<perturbation>` the sensing tests; `sur_only`, `hybrid_1_2`, `hybrid_1_4` the surrogate planners; `policy`, `policy_safety`, `joint_safety` the distilled policy and the safety filter.

Per-episode columns: `viol` (any violation), `v_force`, `v_depth`, `v_pressure`, `v_reflux` (loss-ratio limit exceeded), `success`, `RR` (loss ratio: reflux plus non-target delivery divided by injected volume), `IE` (infusion efficiency), `S` (smooth success score), `Fmax`, `Pmax`, `dmin` (minimum tip-to-RPE distance), `D` (tissue-stress surrogate), `V`, `A`, `Qin`, `En`, `Es`, `aborted`, `t_end`. The variable name `RR` is kept in the code for continuity; it is the loss ratio defined above, not reflux alone.

## Notes on specific quantities

* **Surrogate training set size.** Training data are collected at every fourth planning cycle for all 50 episodes of a batch, with 4 candidates x 4 scenarios x 10 blocks per episode. The runner continues a batch until its last episode ends, and planning cycles are executed (and collected) for all episodes of the batch, including those that have already terminated. Both training batches ran the full 10 s (100 planning cycles), giving 25 x 50 x 160 = 200 000 transitions per batch and 400 000 in total. See `results/logs/surrogate_data_metadata.json`.
* **Training time.** The wall-clock cap of 8 min is checked at the end of each epoch, so the last epoch can exceed it; the full-data model stopped at 8.4 min after 18 epochs.
* **Latency.** `latency.py` measures one procedure at a time (batch size 1). Run it on an otherwise idle machine; concurrent jobs inflate the planner times. Raw per-decision times are stored in `results/logs/latency_raw_ms.npz`.
* **Delay models.** Heavy-tailed jitter is log-normal with median 10 ms and log-space standard deviation 0.8, clipped at 120 ms, applied to actuation and to every sensing channel. The drifting delay is a random walk with 4 ms standard deviation per 20 ms step, bounded to [15, 120] ms.
* **Loss bookkeeping inside the planner.** The planner estimates the volume lost so far as `max(Qin - V_hat, 0)`, which also counts subretinal leakage and is therefore a conservative (upper) estimate.

## License

MIT (see `LICENSE`).
