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

    STATE_VAR_MAP = {
        "DCOILWTICO": "WTI Crude Oil Price",
        "DHHNGSP": "Henry Hub Natural Gas Price",
        "GASREGCOVW": "Gasoline Regular Price",
        "OVXCLS": "Oil Volatility Index (OVX)",
        "XLE_rvol": "XLE Realized Volatility",
    }

    if "params_state_var" in rs.columns:
        rs["params_state_var"] = rs["params_state_var"].replace(STATE_VAR_MAP)


    WEIGHT_VAR_MAP = {
        "binary": "Binary (0 for low-regime, 1 for high-regime)",
        "mean_var": "Mean Variance Optimzied",
    }

    if "params_weight_allocation" in rs.columns:
        rs["params_weight_allocation"] = rs["params_weight_allocation"].replace(WEIGHT_VAR_MAP)

    # --- Relabel parameters nicely ---
    rs = rs.rename(columns={
        "params_state_var": "State Variable",
        "params_weight_allocation": "Weight Allocation Method",
        "params_gamma": "Gamma",
        "params_min_frac": "Outlier Handling"
    })


    # --- Parameter columns to display ---
    param_cols = [
        "State Variable",
        "Weight Allocation Method",
        "Gamma",
        "Outlier Handling"
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
            default_value=default_value, 
            default_opts=default_opts, 
            param_cols=param_cols
        )

    def _register(app: Dash) -> None:
        register_callbacks(app, rs=rs, ts = equity_curves)

    return {
        "title": "Macro Variables",
        "route": "",  # optional for now
        "layout": layout,
        "register_callbacks": _register,
    }
