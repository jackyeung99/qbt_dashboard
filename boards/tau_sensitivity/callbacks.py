# boards/tau_sensitivity/callbacks.py
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback

from .queries import load_equity_all, load_returns_all
from .plots import plot_tau_diagnostics, plot_rv_vs_returns  # or move these to plots.py


def register_callbacks(
    app: Dash,
    *,
    rs: pd.DataFrame,
    param_cols: List[str],
    th_map: Optional[Dict[str, pd.DataFrame]],
) -> None:
    def _empty_fig():
        return px.line(pd.DataFrame({"x": [], "y": []}), x="x", y="y")

    def _fmt(x, kind="float"):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        if kind == "pct":
            return f"{100 * x:.2f}%"
        return f"{x:.3f}"

    def filter_runs(filters: dict) -> pd.DataFrame:
        out = rs
        for k, v in filters.items():
            if v is None:
                continue
            out = out[out[k] == v]
        return out

    dropdown_inputs = [Input({"type": "param-dd", "name": c}, "value") for c in param_cols]

    @callback(
        Output("run-dd", "options"),
        Output("run-dd", "value"),
        Output("filtered-count", "children"),
        dropdown_inputs,
        prevent_initial_call=True,
    )
    def update_run_dropdown(*vals):
        filters = {c: v for c, v in zip(param_cols, vals)}
        dff = filter_runs(filters).sort_values("run_label").head(500)
        opts = [{"label": r["run_label"], "value": r["run_id"]} for r in dff.to_dict("records")]
        default_val = opts[0]["value"] if opts else None
        return opts, default_val, f"{len(dff)} run(s) in list (capped to 500)."

    def plot_equity(run_id: str, strategy: str) -> go.Figure:
        df = load_equity_all()
        fig = go.Figure()

        g = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == "bh")]
        if not g.empty:
            fig.add_trace(go.Scatter(x=g["timestamp"], y=g["equity"], mode="lines",
                                     name="Buy & Hold", line=dict(color="gray", dash="dash")))

        s = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == strategy)]
        if not s.empty and strategy != "bh":
            fig.add_trace(go.Scatter(x=s["timestamp"], y=s["equity"], mode="lines",
                                     name=strategy.upper(), line=dict(width=3)))

        if len(fig.data) == 0:
            fig.add_annotation(text="No equity data", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Equity",
            legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        )
        return fig

    @callback(
        Output("equity-fig", "figure"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-turnover", "children"),
        Output("kpi-tau", "children"),
        Output("stats-table", "data"),
        Output("tau-fig", "figure"),
        Output("tau-help", "children"),
        Output("eval-test", "figure"),
        Output("eval-help", "children"),
        Input("run-dd", "value"),
        Input("strategy-radio", "value"),
        prevent_initial_call=False,
    )
    def render_run(run_id, strategy):
        empty_fig = _empty_fig()

        if not run_id:
            return empty_fig, "-", "-", "-", "-", "-", [], empty_fig, "Select a run.", empty_fig, "Select a run."

        row_df = rs.loc[rs["run_id"] == run_id]
        if row_df.empty:
            return empty_fig, "-", "-", "-", "-", "-", [], empty_fig, "Run not found.", empty_fig, "Run not found."
        row = row_df.iloc[0]

        eq_fig = plot_equity(run_id, strategy)

        prefix = {"c2c": "c2c", "o2c": "o2c", "bh": "bh"}.get(strategy, "c2c")
        sharpe = row.get(f"{prefix}_sharpe", None)
        cumret = row.get(f"{prefix}_cumulative_return", None)
        mdd = row.get(f"{prefix}_max_drawdown", None)
        turnover = row.get(f"{prefix}_turnover", None)
        tau_star = row.get("tau_star", None)

        stats_cols = [
            "intraday_freq", "cutoff", "selection_years", "grid_size", "rf", "transaction_cost",
            "tau_q_lo", "tau_q_hi", "tau_star",
            f"{prefix}_cumulative_return", f"{prefix}_sharpe", f"{prefix}_max_drawdown", f"{prefix}_turnover",
        ]
        stats_cols = [c for c in stats_cols if c in rs.columns]
        stats = [{"metric": c, "value": str(row.get(c))} for c in stats_cols]

        tau_fig = empty_fig
        tau_help = "No τ diagnostics."
        eval_fig = empty_fig
        eval_help = "No RV vs returns data."

        if th_map is not None:
            th_run = th_map.get(run_id)
            if th_run is not None and not th_run.empty and "tau" in th_run.columns:
                tau_fig = plot_tau_diagnostics(th_run, tau_star)
                tau_help = "ΔSharpe over τ grid; dashed line is τ*."

        ret = load_returns_all()
        ret_run = ret[ret["run_id"] == run_id]
        if not ret_run.empty:
            rv_col = "rvol_o2c"
            ret_col = "ret_cc_next" if prefix == "c2c" else "ret_oc_next"
            eval_fig = plot_rv_vs_returns(ret_run, rv_col=rv_col, ret_col=ret_col, tau_star=tau_star)
            eval_help = f"Scatter of {ret_col} vs {rv_col}. Color = split."

        return (
            eq_fig,
            _fmt(sharpe),
            _fmt(cumret, "pct"),
            _fmt(mdd, "pct"),
            _fmt(turnover),
            _fmt(tau_star),
            stats,
            tau_fig,
            tau_help,
            eval_fig,
            eval_help,
        )
