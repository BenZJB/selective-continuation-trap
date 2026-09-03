from __future__ import annotations

from config import ACTUATORS_ZIP, GC_ZIP, METADATA_ZIP, OPLOGS_ZIP, SENSORS_ZIP
from src.zipio import csv_names, read_csv_member


def inspect(name, zip_path, segment=None):
    names = csv_names(zip_path, segment)
    print(f"\n=== {name} ===")
    print(f"CSV members: {len(names)}")
    if names:
        print(f"Example: {names[0]}")
        df = read_csv_member(zip_path, names[0])
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))


def main():
    inspect("Metadata / Operation", METADATA_ZIP, "Operation")
    inspect("Sensors / Operation", SENSORS_ZIP, "Operation")
    inspect("Actuators / Operation", ACTUATORS_ZIP, "Operation")
    inspect("Operation logs", OPLOGS_ZIP)
    inspect("GC / timeseries", GC_ZIP, "timeseries")


if __name__ == "__main__":
    main()
