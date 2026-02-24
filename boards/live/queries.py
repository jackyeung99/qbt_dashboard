from __future__ import annotations
from functools import lru_cache
from typing import List
import pandas as pd
from io import BytesIO
import requests

S3_BASE = "https://quant-trading-project.s3.amazonaws.com/quant-trading/artifacts/live/performance/strategy=StateSignal/universe=XLE"
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