from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# Dashboard repo layout:
# BASE_DIR/
#   results/
#     experiment=<experiment>/
#       runs_summary.parquet
#       timeseries/
#         run_id=.../
#           timeseries.parquet

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = BASE_DIR / "results"


def _experiment_path(experiment: str) -> Path:
    # keep the "experiment=..." partition name
    return RESULTS_ROOT / f"experiment={experiment}"


@lru_cache(maxsize=8)
def load_data(experiment: str = "macro_variables") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    runs_summary : DataFrame
    timeseries   : DataFrame (stacked across runs)
    """
    exp_path = _experiment_path(experiment)

    runs_path = exp_path / "runs_summary.parquet"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs_summary.parquet at: {runs_path}")

    runs_summary = pd.read_parquet(runs_path)

    # Load all run_id timeseries files under:
    # timeseries/run_id=*/timeseries.parquet
    ts_root = exp_path / "timeseries"
    ts_files = sorted(ts_root.rglob("timeseries.parquet")) if ts_root.exists() else []

    ts_list: list[pd.DataFrame] = []
    for f in ts_files:
        df = pd.read_parquet(f)

        # Ensure run_id exists (folder is run_id=<rid>)
        if "run_id" not in df.columns:
            try:
                rid = f.parent.name.split("=", 1)[1]
            except Exception:
                rid = f.parent.name
            df["run_id"] = rid

        ts_list.append(df)

    timeseries = pd.concat(ts_list, axis=0, ignore_index=False) if ts_list else pd.DataFrame()

    return runs_summary, timeseries


@lru_cache(maxsize=8)
def load_timeseries_all(experiment: str = "macro_variables") -> pd.DataFrame:
    """
    Load ALL timeseries once per worker (per experiment).
    Normalizes timestamp + run_id dtype and sorts.
    """
    _, df = load_data(experiment=experiment)

    if df.empty:
        return df

    # normalize timestamp column name if needed
    ts_col = None
    for c in ("timestamp", "date", "datetime", "time"):
        if c in df.columns:
            ts_col = c
            break

    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        sort_cols = ["run_id", ts_col]
    else:
        sort_cols = ["run_id"]

    df["run_id"] = df["run_id"].astype("category")
    return df.sort_values(sort_cols)


def build_pretty_label(row: pd.Series) -> str:
    return f"Run ID = {row.get('run_id','?')}"


def options_from_unique(s: pd.Series) -> List[dict]:
    vals = sorted([v for v in s.dropna().unique().tolist()])
    return [{"label": str(v), "value": v} for v in vals]