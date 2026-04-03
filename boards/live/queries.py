from __future__ import annotations
from functools import lru_cache
from typing import List
import pandas as pd
import numpy as np
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

    # performance = _get(f"{S3_BASE}/portfolio_metrics.json").json()
    meta = _get(f"{S3_BASE}/meta.json").json()

    return equity, meta



def _ensure_session_date(df: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    x = df.copy()
    if date_col in x.columns:
        x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
        x = x.dropna(subset=[date_col]).sort_values(date_col)
    else:
        x.index = pd.to_datetime(x.index, errors="coerce")
        x = x.rename_axis(date_col).reset_index()
        x = x.dropna(subset=[date_col]).sort_values(date_col)

    if not x.empty:
        x = x.drop_duplicates(date_col, keep="last").sort_values(date_col)

    return x


def _coerce_numeric(df: pd.DataFrame, cols: List[str], *, ffill: bool = False) -> pd.DataFrame:
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
            if ffill:
                x[c] = x[c].ffill()
    return x


def _equity_from_returns(r: pd.Series, *, returns_are_log: bool) -> pd.Series:
    r = pd.to_numeric(r, errors="coerce").fillna(0.0).astype(float)
    if returns_are_log:
        return np.exp(r.cumsum())
    return (1.0 + r).cumprod()


def _normalize_to_one(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.empty:
        return s
    s0 = float(s.iloc[0])
    if not np.isfinite(s0) or s0 == 0:
        return pd.Series(np.nan, index=s.index)
    return s / s0


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


def normalize_portfolio(
    equity: pd.DataFrame | pd.Series,
    *,
    date_col: str = "session_date",
    portfolio_value_col: str = "portfolio_value",
    returns_are_log: bool = True,
) -> pd.DataFrame:
    """
    Outputs a DataFrame with:
      - session_date
      - portfolio_value
      - strategy_equity_norm   (from portfolio_value)
      - bh_equity, bh_equity_norm (from SPY returns)
    """

    if equity is None:
        return pd.DataFrame()

    if isinstance(equity, pd.Series):
        equity = equity.to_frame(name=portfolio_value_col)

    x = equity.copy().dropna(subset=['trained_at_utc'])
    x = _ensure_session_date(x, date_col=date_col)

    if portfolio_value_col not in x.columns:
        if "equity" in x.columns:
            x = x.rename(columns={"equity": portfolio_value_col})
        else:
            raise ValueError(f"Missing {portfolio_value_col!r} (and no 'equity' fallback).")

    x = x.sort_values(date_col).reset_index(drop=True)

    initial_value = float(x[portfolio_value_col].iloc[0])

    if "SPY_ret_cc" not in x.columns:
        raise ValueError("Missing 'SPY_ret_cc' column.")

    r = pd.to_numeric(x["SPY_ret_cc"], errors="coerce").fillna(0.0)

    if returns_are_log:
        x["SPY_ret_simple"] = np.exp(r) - 1.0
    else:
        x["SPY_ret_simple"] = r



    # x["bh_equity"] = initial_value * (1.0 + x["SPY_ret_simple"]).cumprod()
    growth = (1.0 + x["SPY_ret_simple"]).cumprod()
    growth[0] = 1.0
    x["bh_equity"] = initial_value * growth

    x["bh_equity_norm"] = x["bh_equity"] / x["bh_equity"].iloc[0]
    x["strategy_equity_norm"] = x[portfolio_value_col] / x[portfolio_value_col].iloc[0]

    return x



def normalize_etf_view(
    df: pd.DataFrame,
    etf: str,
    *,
    meta_cols: list[str] | None = None,
) -> pd.DataFrame:
    if meta_cols is None:
        meta_cols = ["session_date"]

    prefix = f"{etf}_"

    etf_cols = [c for c in df.columns if c.startswith(prefix)]
    if not etf_cols:
        raise ValueError(f"No columns found for ETF {etf!r}")

    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing meta columns: {missing_meta}")

    keep = meta_cols + etf_cols
    out = df[keep].copy()

    renamed = {c: c[len(prefix):] for c in etf_cols}
    out = out.rename(columns=renamed)
  

    if "session_date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"session_date": "date"})
    out = out.sort_values(by=['date'])


    out["weight"] = out["weight"].ffill().fillna(0.0)
    out["_state_var"] = pd.to_numeric(out["rvol"], errors="coerce").ffill().fillna(0)
    out["_tau_star"] = pd.to_numeric(out["_tau_star"], errors="coerce").ffill().fillna(0)

    mask = out["_state_var"].notna() & out["_tau_star"].notna()

    out["signal"] = 0
    out.loc[mask, "signal"] = (out.loc[mask, "_state_var"] > out.loc[mask, "_tau_star"]).astype(int)

    ret = out['ret_cc'].fillna(0.0)
    simple_ret = np.exp(ret) - 1
    out['raw_ret'] = simple_ret

    weight = out["weight"].ffill().fillna(0.0)

    out["etf_ret"] = weight * simple_ret
    out["etf_bh_ret"] = 0.10 * simple_ret

    out["etf_equity"] = (1 + out["etf_ret"]).cumprod()
    out["etf_bh_equity"] = (1 + out["etf_bh_ret"]).cumprod()

    # normalize to start at 1
    if not out.empty:
        out["etf_equity"] /= out["etf_equity"].iloc[0]
        out["etf_bh_equity"] /= out["etf_bh_equity"].iloc[0]

    return out