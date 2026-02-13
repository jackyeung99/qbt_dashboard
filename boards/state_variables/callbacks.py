# boards/tau_sensitivity/callbacks.py
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback
from common.plots import fmt
from .plots import plot_rv_tau_weights_returns_equity_animated


def register_callbacks(
    app: Dash,
    *,
    rs: pd.DataFrame,
    eq: pd.DataFrame,
    param_cols: List[str],
) -> None:
    # ------------------------------------------------------------
    # Precompute indices / maps once (big speedup)
    # ------------------------------------------------------------
    rs_idx = rs.set_index("run_id", drop=False)  # O(1) row lookup

    # groupby once so callbacks don't scan eq each time
    # (dict of run_id -> DF view; we .copy() only when plotting if needed)
    eq_by_run = {rid: g.sort_values("timestamp") for rid, g in eq.groupby("run_id", sort=False)}

    def _empty_fig():
        return px.line(pd.DataFrame({"x": [], "y": []}), x="x", y="y")



    # Faster filtering: build mask on rs in one pass (no repeated slicing)
    def filter_runs(filters: dict) -> pd.DataFrame:
        mask = pd.Series(True, index=rs.index)
        for k, v in filters.items():
            if v is None:
                continue
            mask &= (rs[k] == v)
        return rs.loc[mask]

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

        # to_dict(records) is expensive; iterate rows directly
        opts = [{"label": r.run_label, "value": r.run_id} for r in dff.itertuples(index=False)]
        default_val = opts[0]["value"] if opts else None
        return opts, default_val, f"{len(dff)} run(s) in list (capped to 500)."

    @callback(
        Output("equity-fig", "figure"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-turnover", "children"),
        Output("kpi-tau", "children"),
        Output("stats-table", "data"),
        Input("run-dd", "value"),
        prevent_initial_call=False,
    )
    def render_run(run_id):
        empty_fig = _empty_fig()

        if not run_id:
            return empty_fig, "-", "-", "-", "-", "-", []

        # O(1) lookup instead of rs.loc[rs["run_id"] == run_id]
        try:
            row = rs_idx.loc[run_id]
        except KeyError:
            return empty_fig, "-", "-", "-", "-", "-", [{"metric": "error", "value": "Run not found."}]

        # O(1) lookup instead of eq[eq["run_id"] == run_id]
        dfr = eq_by_run.get(run_id)


        # print(dfr)


        eq_fig = plot_rv_tau_weights_returns_equity_animated(dfr)
        
        # eq_fig = plot_equity_animated(dfr) if dfr is not None and not dfr.empty else empty_fig

        sharpe = row.get("strat_sharpe", None)
        cumret = row.get("strat_cumulative_return", None)
        mdd = row.get("strat_max_drawdown", None)
        turnover = row.get("strat_turnover", None)
        tau_star = row.get("tau_star", None)

    
        stats_cols = [
            "method", "ret_col", "transaction_cost", "state_var",
            "strat_sharpe", "strat_sharpe_buy_regime", "strat_sharpe_no_buy_regime", "strat_sharpe_bh",
            "strat_cumulative_return", "strat_num_buy_days", "strat_num_no_buy_days",
        ]
        stats_cols = [c for c in stats_cols if c in rs.columns]
        stats = [{"metric": c, "value": fmt(row.get(c) ) } for c in stats_cols]

        return (
            eq_fig,
            fmt(sharpe, decimals=2),
            fmt(cumret, style="pct", decimals=2),
            fmt(mdd, style="pct", decimals=2),
            fmt(turnover),
            fmt(tau_star),
            stats,
        )
