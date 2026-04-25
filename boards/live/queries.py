from __future__ import annotations
from functools import lru_cache
from typing import List
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional, Tuple
import requests

S3_ROOT = "https://quant-trading-project.s3.amazonaws.com/quant-trading/artifacts/live/performance"

STRATEGY_BASES = {
    "long_only": f"{S3_ROOT}/strategy=xle-vol-regime-long-only/universe=SPY-Sector",
    "long_short": f"{S3_ROOT}/strategy=xle-vol-regime-long-short/universe=SPY-Sector",
    "sector_long_only": f"{S3_ROOT}/strategy=sector-vol-regime-long-only/universe=SPY-Sector",
    "sector_long_test": f"{S3_ROOT}/strategy=MultiAssetStateSignal/universe=SPY-Sector",
}


def _get(url: str) -> requests.Response:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r


def resolve_s3_base(strategy_key: str) -> str:
    if strategy_key not in STRATEGY_BASES:
        raise ValueError(
            f"Unknown strategy_key={strategy_key!r}. "
            f"Available: {list(STRATEGY_BASES)}"
        )

    return STRATEGY_BASES[strategy_key]


@lru_cache(maxsize=16)
def load_data(
    cache_key: Optional[str] = None,
    strategy_key: str = "long_only",
) -> Tuple[pd.DataFrame, dict]:
    """
    Load dashboard data for the selected strategy.

    cache_key should change daily, e.g. "2026-04-25-long_only",
    so the dashboard refreshes once per day per strategy.
    """

    s3_base = resolve_s3_base(strategy_key)

    eq_resp = _get(f"{s3_base}/portfolio_timeseries.parquet")
    equity = pd.read_parquet(BytesIO(eq_resp.content))

    meta = _get(f"{s3_base}/meta.json").json()

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
    etf: str | None = None,
    is_multi: bool = True,
    *,
    meta_cols: list[str] | None = None,
) -> pd.DataFrame:
    if meta_cols is None:
        meta_cols = ["session_date"]

    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing meta columns: {missing_meta}")

    prefix = f"{etf}_"

    # Multi-asset: select one ETF and strip prefix
    if is_multi and etf is not None:
        etf_cols = [c for c in df.columns if c.startswith(prefix)]

        if not etf_cols:
            raise ValueError(f"No columns found for ETF {etf!r}")

        keep = meta_cols + etf_cols
        out = df[keep].copy()
        out = out.rename(columns={c: c[len(prefix):] for c in etf_cols})

    # Single-asset: keep normal columns and possible XLE_ columns
    else:
        single_asset_cols = [
            "weight",
            "XLE_weight",
            "w_low",
            "XLE_w_low",
            "w_high",
            "XLE_w_high",
            "state_var",
            "XLE_state_var",
            "rvol",
            "XLE_rvol",
            "_state_var",
            "XLE__state_var",
            "_tau_star",
            "XLE__tau_star",
            "tau_star",
            "XLE_tau_star",
            "ret_cc",
            "XLE_ret_cc",
            "raw_ret",
            "XLE_raw_ret",
            "strategy_ret",
            "portfolio_ret",
            "equity",
        ]

        keep = meta_cols + [c for c in single_asset_cols if c in df.columns]
        out = df[keep].copy()

        # Strip XLE_ if standard column does not already exist
        for col in list(out.columns):
            if col.startswith("XLE_"):
                stripped = col[len("XLE_"):]
                if stripped not in out.columns:
                    out = out.rename(columns={col: stripped})

    if "session_date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"session_date": "date"})

    out = out.sort_values(by=["date"])

    # ---- normalize parameter columns ----
    rename_map = {
        "state_var": "_state_var",
        "rvol": "_state_var",
        "tau_star": "_tau_star",
        "w_high": "_w_high",
        "w_low": "_w_low",
    }

    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    if "weight" not in out.columns:
        out["weight"] = 0.0

    if "_state_var" not in out.columns:
        out["_state_var"] = 0.0

    if "_tau_star" not in out.columns:
        out["_tau_star"] = 0.0

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").ffill().fillna(0.0)
    out["_state_var"] = pd.to_numeric(out["_state_var"], errors="coerce").ffill().fillna(0.0)
    out["_tau_star"] = pd.to_numeric(out["_tau_star"], errors="coerce").ffill().fillna(0.0)

    if "_w_high" in out.columns:
        out["_w_high"] = pd.to_numeric(out["_w_high"], errors="coerce").ffill()

    if "_w_low" in out.columns:
        out["_w_low"] = pd.to_numeric(out["_w_low"], errors="coerce").ffill()

    mask = out["_state_var"].notna() & out["_tau_star"].notna()

    out["signal"] = 0
    out.loc[mask, "signal"] = (
        out.loc[mask, "_state_var"] > out.loc[mask, "_tau_star"]
    ).astype(int)

    if "ret_cc" in out.columns:
        simple_ret = np.exp(out["ret_cc"].fillna(0.0)) - 1
    elif "raw_ret" in out.columns:
        simple_ret = out["raw_ret"].fillna(0.0)
    else:
        simple_ret = pd.Series(0.0, index=out.index)

    out["raw_ret"] = simple_ret
    out["etf_ret"] = out["weight"] * simple_ret

    bh_weight = 0.10 if is_multi else 1.0
    out["etf_bh_ret"] = bh_weight * simple_ret

    out["etf_equity"] = (1 + out["etf_ret"]).cumprod()
    out["etf_bh_equity"] = (1 + out["etf_bh_ret"]).cumprod()

    if not out.empty:
        out["etf_equity"] /= out["etf_equity"].iloc[0]
        out["etf_bh_equity"] /= out["etf_bh_equity"].iloc[0]

    return out