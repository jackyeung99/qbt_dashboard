from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, ctx, ALL

from common.plots import fmt
from .plots import plot_rv_tau_weights_returns_equity_animated, plot_rv_tau_weights_returns_equity


def register_callbacks(
    app: Dash,
    *,
    rs: pd.DataFrame,
    ts: pd.DataFrame,
) -> None:
    # ----------------------------
    # Normalize types (prevents silent filter failures)
    # ----------------------------
    rs = rs.copy()
    ts = ts.copy()
    rs["run_id"] = rs["run_id"].astype(str)
    ts["run_id"] = ts["run_id"].astype(str)

    rs_idx = rs.set_index("run_id", drop=False)

    # Prefer sorting by date column if present
    if "date" in ts.columns:
        ts_by_run = {rid: g.sort_values("date") for rid, g in ts.groupby("run_id", sort=False)}
    else:
        ts_by_run = {rid: g.sort_index() for rid, g in ts.groupby("run_id", sort=False)}

    def _empty_fig() -> go.Figure:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        return fig

    # ----------------------------
    # What the filters should use (these are the columns in rs)
    # ----------------------------
    PARAM_COLS = [c for c in ["State Variable", "Weight Allocation Method", "Gamma", "Outlier Handling"] if c in rs.columns]

    PERF_RENAME_MAP = {
        "metric_net_sharpe": "Sharpe",
        "metric_net_cagr": "CAGR",
        "metric_net_max_dd": "Max Drawdown",
        "metric_bh_sharpe": "Sharpe (Buy & Hold)",
        "metric_sharpe_minus_bh": "Sharpe − Buy & Hold",
        "metric_net_ending_equity": "Ending Equity",
    }

    REGIME_RENAME_MAP = {
        "metric_signal_pct_state_1": "% Time Regime Low",
        "metric_signal_pct_state_0": "% Time Regime High",
        "metric_signal_n_turnovers": "Total Turnovers",
        "metric_signal_turnovers_per_year": "Turnovers / Year",
        "metric_signal_avg_hold_state_1": "Avg Low-Regime Duration",
        "metric_signal_avg_hold_state_0": "Avg High-Regime Duration",
    }

    PERF_TABLE_COLS = [c for c in PERF_RENAME_MAP if c in rs.columns]
    REGIME_TABLE_COLS = [c for c in REGIME_RENAME_MAP if c in rs.columns]

    def format_metric_value(col: str, val):
        if pd.isna(val):
            return "-"

        pct_cols = {
            "metric_net_cagr",
            "metric_net_max_dd",
            "metric_signal_pct_state_1",
            "metric_signal_pct_state_0",
        }

        if col in pct_cols:
            print(val)
            return fmt(val/100, style="pct", decimals=2)

        if "sharpe" in col.lower():
            return fmt(val, decimals=2)

        if "duration" in col.lower():
            return fmt(val, decimals=2)

        if "turnover" in col.lower():
            return fmt(val, decimals=2)

        if "equity" in col.lower():
            return fmt(val, decimals=2)

        return fmt(val)
    
    def _rows_from_cols(
        row: pd.Series,
        cols: List[str],
        *,
        key_name: str,
        rename_map: Dict[str, str] | None = None,
    ) -> list[dict]:
        out = []
        for c in cols:
            if c in row.index:
                label = rename_map.get(c, c) if rename_map else c
                out.append({
                    key_name: label,
                    "value": format_metric_value(c, row.get(c)),
                })
        return out


    # ----------------------------
    # Filtering helper (handles None)
    # ----------------------------
    def filter_runs(filters: Dict[str, Any]) -> pd.DataFrame:
        if not filters:
            return rs
        mask = pd.Series(True, index=rs.index)
        for col, v in filters.items():
            if v is None or col not in rs.columns:
                continue
            mask &= (rs[col] == v)
        return rs.loc[mask]

    # ----------------------------
    # Cache heavy plot generation
    # ----------------------------
    @lru_cache(maxsize=128)
    def _cached_fig(run_id: str) -> go.Figure:
        dfr = ts_by_run.get(run_id)
        if dfr is None or dfr.empty:
            return _empty_fig()
        return plot_rv_tau_weights_returns_equity(dfr, run_id=run_id)

    # ============================================================
    # 1) FILTER DROPDOWNS -> RUN DROPDOWN
    # ============================================================
    @callback(
        Output("run-dd", "options"),
        Output("run-dd", "value"),
        Output("filtered-count", "children"),
        Input({"type": "param-dd", "name": ALL}, "value"),
        prevent_initial_call=False,
    )
    def update_run_dropdown(values):
        # Map pattern-matched inputs back to their column name
        meta = ctx.inputs_list[0]  # [{"id": {"type":"param-dd","name":...}, "property":"value"}, ...]
        filters = {m["id"]["name"]: v for m, v in zip(meta, values)}

        dff = filter_runs(filters)

        # Only show runs that have timeseries (avoids selecting dead runs)
        dff = dff[dff["run_id"].isin(ts_by_run.keys())]

        # Build options
        if "run_label" in dff.columns:
            dff = dff.sort_values("run_label")
            opts = [{"label": r.run_label, "value": r.run_id} for r in dff.head(500).itertuples(index=False)]
        else:
            opts = [{"label": rid, "value": rid} for rid in dff["run_id"].head(500).tolist()]

        default_val = opts[0]["value"] if opts else None
        return opts, default_val, f"{len(dff)} run(s) match filters."

    # ============================================================
    # 2) RUN DROPDOWN -> MAIN RENDER
    # ============================================================
    @callback(
        Output("equity-fig", "figure"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-sharpe-diff", "children"),
        Output("kpi-low-regime", "children"),
        Output("kpi-turnover-yr", "children"),
        Output("performance-table", "data"),
        Output("regime-table", "data"),
        Output("chosen-params-table", "data"),
        Input("run-dd", "value"),
        prevent_initial_call=False,
    )
    def render_run(run_id: str):
        if not run_id:
            ef = _empty_fig()
            return ef, "-", "-", "-", "-", "-", "-", [], [], []

        run_id = str(run_id)

        if run_id not in rs_idx.index:
            ef = _empty_fig()
            return ef, "-", "-", "-", "-", "-", "-", [], [], []

        row = rs_idx.loc[run_id]
        fig = _cached_fig(run_id)

        sharpe = row.get("metric_net_sharpe")
        cagr = row.get("metric_net_cagr")
        mdd = row.get("metric_net_max_dd")
        sharpe_diff = row.get("metric_sharpe_minus_bh")
        low_regime = row.get("metric_signal_pct_state_1")
        turnover_yr = row.get("metric_signal_turnovers_per_year")


        performance_rows = _rows_from_cols(
            row,
            PERF_TABLE_COLS,
            key_name="metric",
            rename_map=PERF_RENAME_MAP,
        )

        regime_rows = _rows_from_cols(
            row,
            REGIME_TABLE_COLS,
            key_name="metric",
            rename_map=REGIME_RENAME_MAP,
        )

        chosen_params = _rows_from_cols(
            row,
            PARAM_COLS,
            key_name="param",
        )

        return (
            fig,
            fmt(sharpe, decimals=2),
            fmt(cagr, style="pct", decimals=2),
            fmt(mdd, style="pct", decimals=2),
            fmt(sharpe_diff, decimals=2),
            fmt(low_regime/100, style="pct", decimals=2),
            fmt(turnover_yr, decimals=2),
            performance_rows,
            regime_rows,
            chosen_params,
        )