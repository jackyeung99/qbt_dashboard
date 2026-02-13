from __future__ import annotations

from typing import List
from functools import lru_cache

import pandas as pd
import plotly.graph_objects as go
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
    rs_idx = rs.set_index("run_id", drop=False)
    eq_by_run = {rid: g.sort_values("timestamp") for rid, g in eq.groupby("run_id", sort=False)}

    def _empty_fig():
        # faster than px.line for empty
        fig = go.Figure()
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        return fig

    # ------------------------------------------------------------
    # Pretty labels + aliasing
    # ------------------------------------------------------------
    pretty = {
        "method": "Backtest Method",
        "backtesting_method": "Backtest Method",

        "ret_col": "Trade Timing",
        "execution_type": "Trade Timing",
        "trade_timing": "Trade Timing",

        "weight_type": "Allocation Method",
        "allocation_method": "Allocation Method",

        "state_var": "Volatility Measure",
        "volatility_measure": "Volatility Measure",

        "intraday_freq": "Intraday Frequency",
        "cutoff": "Cutoff Time",
        "selection_years": "Selection Window (years)",
        "grid_size": "τ Grid Size",
        "rf": "Risk-Free Proxy",
        "min_quintile": "τ Quantile Bounds",
        "transaction_cost": "Transaction Cost (bps)",

        "strat_sharpe": "Strategy Sharpe Ratio",
        "strat_sharpe_buy_regime": "Sharpe (Buy Regime)",
        "strat_sharpe_no_buy_regime": "Sharpe (No-Buy Regime)",
        "strat_sharpe_bh": "Buy & Hold Sharpe Ratio",
        "strat_cumulative_return": "Strategy Cumulative Return",
        "strat_num_buy_days": "Buy Regime Days",
        "strat_num_no_buy_days": "No-Buy Regime Days",
    }

    aliases = {
        "method": ["backtesting_method", "method"],
        "ret_col": ["trade_timing", "execution_type", "ret_col"],
        "weight_type": ["allocation_method", "weight_type"],
        "state_var": ["volatility_measure", "state_var"],
    }

    FIXED_COLS = [
        "intraday_freq",
        "cutoff",
        "selection_years",
        "grid_size",
        "rf",
        "min_quintile",
        "transaction_cost",
    ]

    STATS_COLS = [
        "strat_sharpe",
        "strat_sharpe_buy_regime",
        "strat_sharpe_no_buy_regime",
        "strat_sharpe_bh",
        "strat_cumulative_return",
        "strat_num_buy_days",
        "strat_num_no_buy_days",
    ]

    EXTRA_FIXED = [
        {"param": "Cutoff", "value": "4:00 PM ET"},
        {"param": "Realized Variance Intraday Freq", "value": "5 Min"},
    ]

    def _pick_col(row: pd.Series, candidates: List[str]) -> str | None:
        for c in candidates:
            if c in row.index:
                return c
        return None

    def _kv_rows(row: pd.Series, cols: List[str], *, use_aliases: bool = False) -> list[dict]:
        out = []
        for c in cols:
            if c in row.index:
                out.append({"param": pretty.get(c, c), "value": fmt(row.get(c))})
            elif use_aliases and c in aliases:
                picked = _pick_col(row, aliases[c])
                if picked:
                    out.append({"param": pretty.get(picked, picked), "value": fmt(row.get(picked))})
        return out

    def _metric_rows(row: pd.Series, cols: List[str]) -> list[dict]:
        out = []
        for c in cols:
            if c in row.index:
                out.append({"metric": pretty.get(c, c), "value": fmt(row.get(c))})
        return out

    # ------------------------------------------------------------
    # Faster filtering: build mask once; fast-path for no filters
    # ------------------------------------------------------------
    def filter_runs(filters: dict) -> pd.DataFrame:
        if not any(v is not None for v in filters.values()):
            return rs  # no filtering
        mask = pd.Series(True, index=rs.index)
        for k, v in filters.items():
            if v is None or k not in rs.columns:
                continue
            mask &= (rs[k] == v)
        return rs.loc[mask]

    dropdown_inputs = [Input({"type": "param-dd", "name": c}, "value") for c in param_cols]

    # ------------------------------------------------------------
    # Cache heavy plot generation (HUGE speedup)
    # ------------------------------------------------------------
    @lru_cache(maxsize=64)
    def _cached_fig(run_id: str) -> go.Figure:
        dfr = eq_by_run.get(run_id)
        if dfr is None or dfr.empty:
            return _empty_fig()
        # IMPORTANT: dfr is already sorted in eq_by_run
        return plot_rv_tau_weights_returns_equity_animated(dfr, run_id=run_id)

    # ------------------------------------------------------------
    # Run dropdown updates
    # ------------------------------------------------------------
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

        opts = [{"label": r.run_label, "value": r.run_id} for r in dff.itertuples(index=False)]
        default_val = opts[0]["value"] if opts else None
        return opts, default_val, f"{len(dff)} run(s) in list (capped to 500)."

    # ------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------
    @callback(
        Output("equity-fig", "figure"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-turnover", "children"),
        Output("kpi-tau", "children"),
        Output("chosen-params-table", "data"),
        Output("fixed-settings-table", "data"),
        Output("stats-table", "data"),
        Input("run-dd", "value"),
        prevent_initial_call=False,
    )
    def render_run(run_id):
        if not run_id:
            ef = _empty_fig()
            return ef, "-", "-", "-", "-", "-", [], [], []

        # O(1) lookup
        try:
            row = rs_idx.loc[run_id]
        except KeyError:
            ef = _empty_fig()
            return ef, "-", "-", "-", "-", "-", [], [], [{"metric": "Error", "value": "Run not found."}]

        # cached heavy fig
        eq_fig = _cached_fig(run_id)

        # KPIs (cheap)
        sharpe = row.get("strat_sharpe", None)
        cumret = row.get("strat_cumulative_return", None)
        mdd = row.get("strat_max_drawdown", None)
        turnover = row.get("strat_turnover", None)
        tau_star = row.get("tau_star", None)

        # tables (cheap)
        chosen_rows = _kv_rows(row, param_cols, use_aliases=True)
        fixed_rows = _kv_rows(row, FIXED_COLS, use_aliases=False)
        fixed_rows.extend(EXTRA_FIXED)

        stats_rows = _metric_rows(row, STATS_COLS)

        return (
            eq_fig,
            fmt(sharpe, decimals=2),
            fmt(cumret, style="pct", decimals=2),
            fmt(mdd, style="pct", decimals=2),
            fmt(turnover),
            fmt(tau_star),
            chosen_rows,
            fixed_rows,
            stats_rows,
        )
