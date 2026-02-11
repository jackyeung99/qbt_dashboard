from __future__ import annotations
from functools import lru_cache

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS = BASE_DIR / "results" / "xle_rv_sweep"



@lru_cache(maxsize=1)
def load_data():
    runs_summary = pd.read_parquet(RESULTS / "runs_summary.parquet")
    # equity_curves = pd.read_parquet(RESULTS / "equity_curves.parquet")
    thresholds = pd.read_parquet(RESULTS / "thresholds.parquet")
    # returns = pd.read_parquet(RESULTS / "returns.parquet")[['run_id', 'timestamp', 'split', 'rvol_o2c', 'ret_cc', 'ret_oc']]
    return runs_summary, thresholds



@lru_cache(maxsize=1)
def load_equity_all() -> pd.DataFrame:
    """Load ALL equity curves once per worker."""
    df = pd.read_parquet(
        RESULTS / "equity_curves.parquet",
        columns=["run_id", "strategy", "timestamp", "equity"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["run_id"] = df["run_id"].astype("category")
    df["strategy"] = df["strategy"].astype("category")
    return df.sort_values(["run_id", "strategy", "timestamp"])


@lru_cache(maxsize=1)
def load_returns_all() -> pd.DataFrame:
    """Load ALL returns once per worker (slice by run_id in memory)."""
    cols = ["run_id", "timestamp", "split", "rvol_o2c", "ret_cc", "ret_oc", "ret_cc_next", "ret_oc_next"]
    df = pd.read_parquet(RESULTS / "returns.parquet", columns=cols)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["run_id"] = df["run_id"].astype("category")
    if "split" in df.columns:
        df["split"] = df["split"].astype("category")
    return df.sort_values(["run_id", "timestamp"])


def build_pretty_label(row: pd.Series) -> str:
    parts = [
        f"{row.get('intraday_freq','?')}",
        f"cutoff={row.get('cutoff','?')}",
        f"yrs={row.get('selection_years','?')}",
        f"grid={row.get('grid_size','?')}",
        f"rf={row.get('rf','?')}",
        f"tc={row.get('transaction_cost','?')}bps",
        f"τ*={row.get('tau_star', float('nan')):.3g}" if pd.notna(row.get("tau_star", None)) else "τ*=?",
        f"Sh(c2c)={row.get('c2c_sharpe', float('nan')):.2f}" if pd.notna(row.get("c2c_sharpe", None)) else "Sh=?",
    ]
    return " | ".join(parts)


def options_from_unique(s: pd.Series) -> List[dict]:
    vals = sorted([v for v in s.dropna().unique().tolist()])
    return [{"label": str(v), "value": v} for v in vals]


def make_threshold_map(thresholds: Optional[pd.DataFrame]) -> Optional[Dict[str, pd.DataFrame]]:
    if thresholds is None:
        return None
    th = thresholds.copy()
    if "run_id" not in th.columns:
        raise ValueError("thresholds must contain 'run_id'")
    return {rid: g for rid, g in th.groupby("run_id", sort=False)}
