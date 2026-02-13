# boards/tau_sensitivity/dashboard.py
from __future__ import annotations

import pandas as pd
from dash import Dash
from dash.development.base_component import Component

from .queries import load_data, build_pretty_label
from .layout import build_layout
from .callbacks import register_callbacks


def build_dashboard(ctx):
    """
    Registry entrypoint.
    Returns a dict with:
      - title
      - route (optional)
      - layout() -> Dash Component
      - register_callbacks(app) -> None
    """
    runs_summary, equity_curves = load_data()

    rs = runs_summary.copy()
    if "run_id" not in rs.columns:
        raise ValueError("runs_summary must contain 'run_id'")

    if "run_label" not in rs.columns:
        rs["run_label"] = rs.apply(build_pretty_label, axis=1)



    # --- Relabel parameters nicely ---
    
    rs = rs.rename(columns={
        "method": "backtesting_method",
        "ret_col": "trade_timing",
        "weight_type": "allocation_method",
        "state_var": "volatility_measure"
    })
    
    if "allocation_method" in rs.columns:
        rs["allocation_method"] = rs["allocation_method"].replace({
            "fixed": "Fixed (1 in low regime, 0 in high regime)",
            "mean_var": "Mean-variance optimized weights",
        })

    
    if "trade_timing" in rs.columns:
        rs["trade_timing"] = rs["trade_timing"].replace({
            "ret_cc": "Close-to-Close (C→C)",
            "ret_oc": "Open-to-Close (O→C)",
            "ret_co": "Close-to-Open (C→O)",
            "ret_oo": "Open-to-Open (O→O)",
        })

    if "volatility_measure" in rs.columns:
        rs["volatility_measure"] = rs["volatility_measure"].replace({
            "rvol":    "Realized Volatility (RV, observed at t)",
            "ewma":    "EWMA Next-Day Volatility Forecast",
            "har_rv":  "HAR-RV Next-Day Volatility Forecast",
            "garch11": "GARCH(1,1) Next-Day Volatility Forecast",
        })
    # --- Parameter columns to display ---
    param_cols = [
        "backtesting_method",
        "volatility_measure",
        "trade_timing",
        "allocation_method",
    ]
    param_cols = [c for c in param_cols if c in rs.columns]


    # --- Existing logic (unchanged) ---
    rs_sorted = rs.sort_values("run_label")
    rs_default = rs_sorted.head(200)

    default_opts = [
        {"label": r["run_label"], "value": r["run_id"]}
        for r in rs_default.to_dict("records")
    ]

    default_value = default_opts[0]["value"] if default_opts else None

    def layout() -> Component:
        return build_layout(
            runs_summary=rs,
            param_cols=param_cols,
            default_opts=default_opts,
            default_value=default_value,
        )

    def _register(app: Dash) -> None:
        register_callbacks(app, rs=rs, eq = equity_curves, param_cols=param_cols)

    return {
        "title": "State Variables",
        "route": "",  # optional for now
        "layout": layout,
        "register_callbacks": _register,
    }
