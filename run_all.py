from __future__ import annotations

from src.inspect_data import main as inspect_main
from src.build_run_table import build_run_table
from src.define_endpoint import build_endpoints
from src.make_prefix_features import build_prefix_features
from experiments import endpoint_check, prefix_predictability, late_bloomers, censoring_replay


def main():
    steps = [
        ("inspect raw archives", inspect_main),
        ("build run table", build_run_table),
        ("build endpoints", build_endpoints),
        ("build prefix features", build_prefix_features),
        ("endpoint check", endpoint_check.main),
        ("prefix predictability", prefix_predictability.main),
        ("late bloomers", late_bloomers.main),
        ("quick censoring replay", censoring_replay.main),
    ]

    for name, func in steps:
        print("\n" + "=" * 80)
        print(f"RUNNING {name}")
        print("=" * 80)
        result = func()
        if hasattr(result, "shape"):
            print(f"Completed: output shape {result.shape}")

    print("\nDone. Inspect results/tables and results/figures.")
    print("\nNext confirmatory experiment:")
    print("  python run_robustness.py")


if __name__ == "__main__":
    main()
