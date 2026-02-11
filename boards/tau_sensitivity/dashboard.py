# boards/tau_sensitivity/dashboard.py
from __future__ import annotations

import pandas as pd
from dash import Dash
from dash.development.base_component import Component

from .queries import load_data, build_pretty_label, make_threshold_map
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
    runs_summary, thresholds = load_data()

    rs = runs_summary.copy()
    if "run_id" not in rs.columns:
        raise ValueError("runs_summary must contain 'run_id'")

    if "run_label" not in rs.columns:
        rs["run_label"] = rs.apply(build_pretty_label, axis=1)

    param_cols = [
        "intraday_freq", "cutoff", "selection_years", "grid_size", "rf",
        "transaction_cost", "tau_quantile_bounds",
    ]
    param_cols = [c for c in param_cols if c in rs.columns]

    th_map = make_threshold_map(thresholds)

    rs_sorted = rs.sort_values("run_label")
    rs_default = rs_sorted.head(200)
    default_opts = [{"label": r["run_label"], "value": r["run_id"]} for r in rs_default.to_dict("records")]
    default_value = default_opts[0]["value"] if default_opts else None

    def layout() -> Component:
        return build_layout(
            runs_summary=rs,
            param_cols=param_cols,
            default_opts=default_opts,
            default_value=default_value,
        )

    def _register(app: Dash) -> None:
        register_callbacks(app, rs=rs, param_cols=param_cols, th_map=th_map)

    return {
        "title": "Tau Sensitivity",
        "route": "/tau",  # optional for now
        "layout": layout,
        "register_callbacks": _register,
    }
