from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

METADATA_ZIP = RAW_DIR / "00_metadata.zip"
SENSORS_ZIP = RAW_DIR / "01_sensors.zip"
ACTUATORS_ZIP = RAW_DIR / "02_actuators.zip"
OPLOGS_ZIP = RAW_DIR / "07_operation_logs.zip"
GC_ZIP = RAW_DIR / "10_gc_composition.zip"
PLANT_YAML = RAW_DIR / "plant_description.yaml"

PRIMARY_MIXTURE = "batch_dist_ternary_butan-1-ol+propan-2-ol+water"
PREFIXES = (0.10, 0.20, 0.40)
RANDOM_SEED = 42
N_CV_SPLITS = 5

GOOD_QUANTILE = 0.80
POOR_PRED_QUANTILE = 0.30
STOP_QUANTILES = (0.20, 0.30, 0.40)
REPLAY_SEEDS = tuple(range(5))
INITIAL_TRAIN_FRACTION = 0.40
TARGET_ROUND_RUNS = 8

# Confirmatory robustness sweep.
ROBUST_N_SEEDS = int(os.getenv("TRAP_N_SEEDS", "50"))
ROBUST_REPLAY_SEEDS = tuple(range(ROBUST_N_SEEDS))
ROBUST_STOP_QUANTILES = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60,
)
ROBUST_PREFIX = 0.20
ROBUST_INTERVAL_LEVEL = 0.95
ROBUST_BOOTSTRAP_REPS = int(os.getenv("TRAP_BOOTSTRAP_REPS", "2000"))
ROBUST_BOOTSTRAP_SEED = 20260903

for d in (PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
