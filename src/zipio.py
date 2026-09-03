from __future__ import annotations

from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import PurePosixPath
from typing import Iterable
import zipfile

import pandas as pd


@dataclass(frozen=True)
class RunKey:
    mixture: str
    operating_point: str
    experiment: str

    @property
    def run_id(self) -> str:
        return f"{self.mixture}__{self.operating_point}__{self.experiment}"


def csv_names(zip_path, required_segment: str | None = None) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if required_segment:
        token = f"/{required_segment.strip('/')}/"
        names = [n for n in names if token in f"/{n}"]
    return sorted(names)


def parse_run_key(member_name: str) -> RunKey:
    """Parse any dataset member ending mixture/operating_point_xxx/experiment.csv."""
    p = PurePosixPath(member_name)
    parts = p.parts
    if len(parts) < 3:
        raise ValueError(f"Cannot parse run key from {member_name}")
    mixture, operating_point, filename = parts[-3], parts[-2], parts[-1]
    return RunKey(mixture, operating_point, PurePosixPath(filename).stem)


def build_member_index(zip_path, required_segment: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for member in csv_names(zip_path, required_segment=required_segment):
        key = parse_run_key(member)
        out[key.run_id] = member
    return out


def read_csv_member(zip_path, member_name: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as fh:
            return pd.read_csv(fh, **kwargs)


def list_members(zip_path, suffix: str | None = None) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    if suffix:
        names = [n for n in names if n.lower().endswith(suffix.lower())]
    return names
