# Selective Continuation Trap — real batch-distillation replay

This project tests whether autonomous early stopping creates policy-induced endpoint
censoring that becomes self-reinforcing under completed-only retraining.

## Standard pipeline

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py
```

## Next confirmatory experiment

Run:

```powershell
python run_robustness.py
```

Defaults:
- 20% trajectory prefix;
- 50 deployment orderings;
- stop-threshold quantiles 0.10, 0.15, ..., 0.60;
- paired completed-only vs oracle comparisons.

### Quick smoke test

```powershell
$env:TRAP_N_SEEDS="3"
$env:TRAP_BOOTSTRAP_REPS="200"
python run_robustness.py
```

Then clear the overrides before the real run:

```powershell
Remove-Item Env:TRAP_N_SEEDS -ErrorAction SilentlyContinue
Remove-Item Env:TRAP_BOOTSTRAP_REPS -ErrorAction SilentlyContinue
python run_robustness.py
```

For 100 deployment orderings:

```powershell
$env:TRAP_N_SEEDS="100"
python run_robustness.py
```

## New outputs

Tables:
- `05_robustness_decisions.csv`
- `05_round_metrics.csv`
- `05_seed_level_summary.csv`
- `05_paired_contrasts_by_seed.csv`
- `05_stop_sweep_summary.csv`
- `05_cumulative_round_metrics.csv`
- `05_dose_response_by_seed.csv`

Figures:
- `05_recall_gap_vs_stop_quantile.png`
- `05_false_stop_gap_vs_stop_quantile.png`
- `05_cumulative_recall_q40.png`
- `05_recall_gap_vs_actual_stop_rate.png`

## Sign convention

All paired gaps are:

`completed-only − oracle`.

Therefore:
- recall gap < 0 = censoring is worse;
- false-stop gap > 0 = censoring is worse;
- regret gap > 0 = censoring is worse;
- MAE gap > 0 = censoring is worse;
- Spearman gap < 0 = censoring is worse.

A convincing trap should be robust across deployment orderings and should preferably
become stronger as stopping gets more aggressive.

## Statistical caution

The seed sweep reorders the same physical experiments. Its intervals therefore measure
robustness to deployment ordering, not population-level uncertainty over new experiments.
