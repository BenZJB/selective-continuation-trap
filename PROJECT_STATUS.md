# Current project status

## Latest successful baseline run

The user's latest run produced:

### Prefix predictability

| Prefix | MAE | RMSE | Spearman |
|---|---:|---:|---:|
| 10% | 0.059754 | 0.093155 | 0.278045 |
| 20% | 0.056457 | 0.090917 | 0.311396 |
| 40% | 0.058533 | 0.088761 | 0.300407 |

### Exploratory late-bloomer candidates

| Prefix | Valuable runs | Late-bloomer candidates | Fraction |
|---|---:|---:|---:|
| 10% | 14 | 5 | 35.7% |
| 20% | 14 | 5 | 35.7% |
| 40% | 14 | 6 | 42.9% |

### Quick censoring replay

Valuable-run recall:

| Stop quantile | Completed-only | Oracle |
|---|---:|---:|
| 0.20 | 0.714286 | 0.752381 |
| 0.30 | 0.692857 | 0.716667 |
| 0.40 | 0.692857 | 0.716667 |

The completed-only learner is worse in the expected direction at all three tested
thresholds in this run. This is promising but not yet sufficient to claim a general,
self-reinforcing Selective-Continuation Trap.

## Current gate

The next task is to determine whether the gap is robust to:
1. many deployment orderings;
2. a fine sweep of stopping aggressiveness;
3. cumulative deployment rounds.

Run:

```powershell
python run_robustness.py
```

A strong result would show:
- completed-only recall gap < 0 across most thresholds/orderings;
- completed-only false-stop gap > 0;
- increasing harm with more aggressive stopping;
- progressive cumulative separation from oracle over rounds.

Only after that should JEPA/Flow/auditing become the main implementation focus.
