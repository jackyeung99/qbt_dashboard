from __future__ import annotations

from typing import Any, Mapping
from dash import Dash, Output, Input, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from common.plots import fmt
from .plots import *
from common.helpers import format_et
from common.metrics import _perf_metrics, _signal_metrics, compute_portfolio_metrics
from .queries import *



def build_etf_view_model(etf_df: pd.DataFrame, selected_etf: str) -> dict:
    sig_metrics = _signal_metrics(etf_df["signal"], ann_factor=252)
    perf_metrics = _perf_metrics(
        etf_df["etf_ret"],
        ann_factor=252,
        return_type="simple",
        prefix="",
        initial_equity=100_000,
    )

    current_signal = int(etf_df["signal"].iloc[-1]) if not etf_df.empty else 0
    current_regime = (
        "High-Volatility Regime" if current_signal == 1 else "Low-Volatility Regime"
    )

    return {
        "etf": selected_etf,
        "kpis": {
            "return": perf_metrics.get("total_pnl"),
            "sharpe": perf_metrics.get("sharpe"),
            "regime": current_regime,
            "weight": etf_df["weight"].iloc[-1] if "weight" in etf_df else np.nan,
        },
        "signal_table": [
            {"metric": "Current Regime", "value": current_regime},
            {"metric": "Days in Current Regime", "value": sig_metrics.get("days_in_current_regime")},
            {"metric": "% Time High Regime", "value": sig_metrics.get("pct_high_regime")},
            {"metric": "% Time Low Regime", "value": sig_metrics.get("pct_low_regime")},
            {"metric": "Signal Flips", "value": sig_metrics.get("n_flips")},
            {"metric": "Avg Regime Duration", "value": sig_metrics.get("avg_regime_duration")},
        ],
    }

# -----------------------
# Dash callback wiring
# -----------------------
def register_callbacks(app: Dash, *, load_live_data) -> None:
    @callback(
        Output("equity-fig", "figure"),
        Output("allocation-fig", "figure"),
        Output("generated_at", "children"),
        Output("kpi-total-pnl", "children"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-mean-return", "children"),
        Output("kpi-return-vol", "children"),
        Output("portfolio-stats-panel", "children"),
        Output("weights-stats-panel", "figure"),
        Input("url", "pathname"),
        prevent_initial_call=False,
    )
    def render_live(_pathname):
        try:
            equity_raw, meta = load_live_data()
            equity = normalize_portfolio(equity_raw)

            if equity is None or equity.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(template="plotly_white")

                return (
                    empty_fig,
                    empty_fig,
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    html.Div("No portfolio statistics available."),
                    html.Div("No weight statistics available."),
                )

            fig = plot_portfolio(equity)
            allocation_fig = plot_allocation_pie(equity)

            k = compute_portfolio_metrics(equity)

       
            portfolio_table = build_portfolio_table_card(k)
            weights_table = plot_avg_weights_from_metrics(k)
    

            generated_at = "-"
            if meta and isinstance(meta, dict) and meta.get("generated_at_utc"):
                generated_at = format_et(meta["generated_at_utc"])

            return (
                fig,
                allocation_fig,
                generated_at,
                fmt(k.get("total_pnl"), decimals=2),
                fmt(k.get("sharpe"), decimals=2),
                fmt(k.get("cagr"), style="pct", decimals=2),
                fmt(k.get("max_dd"), style="pct", decimals=2),
                fmt(k.get("mean_ann"), style="pct", decimals=2),
                fmt(k.get("vol_ann"), style="pct", decimals=2),
                portfolio_table,
                weights_table,
            )

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Load error: {e}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(template="plotly_white")

            err_panel = html.Div(
                f"Load error: {e}",
                style={"color": "#b91c1c", "fontWeight": 600},
            )

            return (
                fig,
                fig,
                "ERR",
                "ERR",
                "ERR",
                "ERR",
                "ERR",
                "ERR",
                "ERR",
                err_panel,
                err_panel,
            )





    @callback(
        Output("etf-main-fig", "figure"),
        Output("etf-kpi-return", "children"),
        Output("etf-kpi-sharpe", "children"),
        Output("etf-kpi-regime", "children"),
        Output("etf-kpi-weight", "children"),
        Output("etf-side-panel", "children"),
        Input("etf-selector", "value"),
    )
    def update_etf_drilldown(selected_etf):
        equity_raw, _ = load_live_data()
        equity = normalize_portfolio(equity_raw)

        if equity is None or equity.empty or selected_etf is None:
            return {}, "-", "-", "-", "-", html.Div("No data")

        etf_df = normalize_etf_view(equity, selected_etf)

        sig_metrics = _signal_metrics(etf_df["signal"], ann_factor=252)
        perf_metrics = _perf_metrics(
            etf_df["etf_ret"],
            ann_factor=252,
            return_type="simple",
            prefix="",
            initial_equity=100_000,
        )

        fig = plot_rv_tau_weights_returns_equity(etf_df, etf=selected_etf)


        cur_regime = (
            "High-Volatility Regime"
            if int(etf_df["signal"].iloc[-1]) == 1
            else "Low-Volatility Regime"
        )

        side_panel = build_signal_table_card(etf_df, sig_metrics)

        return (
            fig,
            fmt(perf_metrics.get("total_return"), style="pct", decimals=2),
            fmt(perf_metrics.get("sharpe"), decimals=2),
            cur_regime,
            fmt(etf_df["weight"].iloc[-1], style="pct", decimals=1),
            side_panel,
        )