from __future__ import annotations

from typing import Any, Mapping
from dash import Dash, Output, Input, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from common.plots import fmt
from .plots import normalize_equity_df, plot_rv_tau_weights_returns_equity_animated
from common.helpers import format_et


# -----------------------
# Helpers
# -----------------------
def pick_metric(d: Mapping[str, Any], *keys: str, default=np.nan):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default



def compute_kpis(
    equity: pd.DataFrame,
    *,
    performance: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    ann_factor: int = 252,
) -> dict:
    performance = dict(performance or {})
    meta = dict(meta or {})

    # Prefer performance.json KPIs (flexible key names)
    sharpe_p = pick_metric(performance, "sharpe", "sharpe_ratio", "strat_sharpe")
    cagr_p = pick_metric(performance, "cagr", "CAGR", "ann_return", "annualized_return")
    mdd_p = pick_metric(performance, "max_drawdown", "mdd", "maxDD", "strat_max_drawdown")
    mean_returns = pick_metric(performance, "mean_daily")
    return_vol = pick_metric(performance, "vol_daily",)

    tau_star = None

    if tau_star is None or (isinstance(tau_star, float) and np.isnan(tau_star)):
        tau_star = pick_metric(meta, "tau_star", default=np.nan)

    # Fallbacks from equity series if missing
    pv = equity["portfolio_value"].astype(float)

    # Sharpe fallback
    if (sharpe_p is None) or (not np.isfinite(float(sharpe_p))):
        r = pv.pct_change().dropna()
        sd = float(r.std(ddof=1))
        sharpe = (float(r.mean()) / sd) * np.sqrt(ann_factor) if sd > 0 else np.nan
    else:
        sharpe = float(sharpe_p)

    # CAGR fallback
    if (cagr_p is None) or (not np.isfinite(float(cagr_p))):
        t0, t1 = equity["session_date"].iloc[0], equity["session_date"].iloc[-1]
        years = max((t1 - t0).days / 365.25, 1e-9)
        cagr = float((pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1) if pv.iloc[0] > 0 else np.nan
    else:
        cagr = float(cagr_p)

    # MDD fallback
    if (mdd_p is None) or (not np.isfinite(float(mdd_p))):
        running_max = pv.cummax()
        mdd = float((pv / running_max - 1).min())
    else:
        mdd = float(mdd_p)


    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "mdd": mdd,
        "mean_return": mean_returns,
        "return_vol": return_vol,
    }


# -----------------------
# Dash callback wiring
# -----------------------
def register_callbacks(app: Dash, *, load_live_data) -> None:
    @callback(
        Output("equity-fig", "figure"),
        Output("generated_at", "children"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-mean-return", "children"),
        Output("kpi-return-vol", "children"),
        Input("url", "pathname"),
        prevent_initial_call=False,
    )
    def render_live(_pathname):
        try:
            equity_raw, performance, meta = load_live_data()
            # print(equity_raw.head())
            # print(equity_raw.tail())

        
            # normalize + plot
            equity = normalize_equity_df(equity_raw)


            fig = plot_rv_tau_weights_returns_equity_animated(equity)

            if equity.empty:
                return fig, "-", "-", "-", "-", "-"

            # KPIs
            k = compute_kpis(equity, performance=performance, meta=meta)

            return (
                fig,
                format_et(meta['generated_at_utc']),
                fmt(k["sharpe"], decimals=2),
                fmt(k["cagr"], style="pct", decimals=2),
                fmt(k["mdd"], style="pct", decimals=2),
                fmt(k["mean_return"]),
                fmt(k["return_vol"]),
            )

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Load error: {e}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(template="plotly_white")
            return fig, "ERR", "ERR", "ERR", "ERR", "ERR"
