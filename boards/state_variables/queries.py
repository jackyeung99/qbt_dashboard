from __future__ import annotations
from functools import lru_cache

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS = BASE_DIR / "results" / "state_pred_sweep"



@lru_cache(maxsize=1)
def load_data():
    runs_summary = pd.read_parquet(RESULTS / "runs_summary.parquet")
    equity_curves = pd.read_parquet(RESULTS / "equity_curves.parquet")
    # thresholds = pd.read_parquet(RESULTS / "thresholds.parquet")
    # returns = pd.read_parquet(RESULTS / "returns.parquet")[['run_id', 'timestamp', 'split', 'rvol_o2c', 'ret_cc', 'ret_oc']]
    return runs_summary, equity_curves



@lru_cache(maxsize=1)
def load_equity_all() -> pd.DataFrame:
    """Load ALL equity curves once per worker."""
    df = pd.read_parquet(
        RESULTS / "equity_curves.parquet",
        # columns=["run_id", "strategy", "timestamp", "equity", "turnover"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["run_id"] = df["run_id"].astype("category")
    return df.sort_values(["run_id","timestamp"])


def build_pretty_label(row: pd.Series) -> str:
    parts = [
        f"Run ID = {row.get('run_id','?')}",
    ]
    return " | ".join(parts)


def options_from_unique(s: pd.Series) -> List[dict]:
    vals = sorted([v for v in s.dropna().unique().tolist()])
    return [{"label": str(v), "value": v} for v in vals]


