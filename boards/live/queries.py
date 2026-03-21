from __future__ import annotations
from functools import lru_cache
from typing import List
import pandas as pd
from io import BytesIO
import requests

S3_BASE = "https://quant-trading-project.s3.amazonaws.com/quant-trading/artifacts/live/performance/strategy=MultiAssetStateSignal/universe=SPY-Sector"
def _get(url: str) -> requests.Response:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r



@lru_cache(maxsize=1)
def load_data(_key=None):
   
    eq_resp = _get(f"{S3_BASE}/portfolio_timeseries.parquet")
    equity = pd.read_parquet(BytesIO(eq_resp.content))


    performance = _get(f"{S3_BASE}/portfolio_metrics.json").json()
    meta = _get(f"{S3_BASE}/meta.json").json()

    return equity, performance, meta


def extract_etfs_from_weights(df) -> list[str]:
    return sorted(
        col[:-len("_weight")]
        for col in df.columns
        if col.endswith("_weight")
    )

def pick_default_etf(etfs: list[str]) -> str | None:
    if not etfs:
        return None
    if "XLE" in etfs:
        return "XLE"
    return etfs[0]


def pick_etf_columns(etf: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(f"{etf}_") or c in ['session_date']]
    return df[cols]