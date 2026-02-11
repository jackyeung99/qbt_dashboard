# src/dashboards/boards/tau_sensitivity/layout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from dash import dcc, dash_table, html

from .queries import options_from_unique
# from .meta import TITLE


def kpi_card(title: str, value_id: str) -> html.Div:
    return html.Div(
        style={
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "padding": "10px 12px",
            "minWidth": 0,
            "background": "white",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": "#666"}),
            html.Div(id=value_id, style={"fontSize": "22px", "fontWeight": "600"}),
        ],
    )


def build_layout(*, runs_summary: pd.DataFrame, param_cols: List[str], default_opts: List[dict], default_value):
    return html.Div(
        style={"fontFamily": "system-ui", "padding": "16px", "background": "#f6f7fb"},
        children=[
            html.H2("Tau Sensitivity", style={"margin": "0 0 10px 0"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "12px", "minWidth": 0},
                children=[
                    # =========================
                    # Left: Filters
                    # =========================
                    html.Div(
                        style={"background": "white", "borderRadius": "12px", "padding": "12px", "border": "1px solid #e6e6e6"},
                        children=[
                            html.Div("Filters", style={"fontWeight": 700, "marginBottom": "8px"}),
                            *[
                                html.Div(
                                    style={"marginBottom": "10px"},
                                    children=[
                                        html.Div(c, style={"fontSize": "12px", "color": "#666"}),
                                        dcc.Dropdown(
                                            id={"type": "param-dd", "name": c},
                                            options=options_from_unique(runs_summary[c]),
                                            value=None,
                                            placeholder=f"All {c}",
                                            clearable=True,
                                        ),
                                    ],
                                )
                                for c in param_cols
                            ],
                            html.Hr(),
                            html.Div("Strategy", style={"fontSize": "12px", "color": "#666"}),
                            dcc.RadioItems(
                                id="strategy-radio",
                                options=[{"label": x, "value": x} for x in ["c2c", "o2c"]],
                                value="c2c",
                                inline=True,
                                style={"marginBottom": "12px"},
                            ),
                            html.Div("Run", style={"fontSize": "12px", "color": "#666"}),
                            dcc.Dropdown(
                                id="run-dd",
                                options=default_opts,
                                value=default_value,
                                placeholder="Select a run",
                                clearable=False,
                            ),
                            html.Div(id="filtered-count", style={"marginTop": "10px", "fontSize": "12px", "color": "#666"}),
                        ],
                    ),
                    # =========================
                    # Right: Content
                    # =========================
                    html.Div(
                        style={"minWidth": 0},
                        children=[
                            html.Div(
                                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px", "minWidth": 0},
                                children=[
                                    kpi_card("Sharpe", "kpi-sharpe"),
                                    kpi_card("CAGR", "kpi-cagr"),
                                    kpi_card("Max Drawdown", "kpi-mdd"),
                                    kpi_card("Turnover", "kpi-turnover"),
                                    kpi_card("τ*", "kpi-tau"),
                                ],
                            ),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 420px", "gap": "12px", "minWidth": 0},
                                children=[
                                    html.Div(
                                        style={
                                            "background": "white",
                                            "borderRadius": "12px",
                                            "padding": "12px",
                                            "border": "1px solid #e6e6e6",
                                            "minWidth": 0,
                                        },
                                        children=[
                                            html.Div("Equity Curve", style={"fontWeight": 700, "marginBottom": "6px"}),
                                            dcc.Graph(
                                                id="equity-fig",
                                                config={"displayModeBar": False, "responsive": True},
                                                style={"height": "420px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "background": "white",
                                            "borderRadius": "12px",
                                            "padding": "12px",
                                            "border": "1px solid #e6e6e6",
                                            "minWidth": 0,
                                        },
                                        children=[
                                            html.Div("Run Stats", style={"fontWeight": 700, "marginBottom": "6px"}),
                                            dash_table.DataTable(
                                                id="stats-table",
                                                columns=[{"name": "metric", "id": "metric"}, {"name": "value", "id": "value"}],
                                                data=[],
                                                style_cell={"fontSize": 12, "padding": "6px"},
                                                style_header={"fontWeight": "700", "background": "#fafafa"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "marginTop": "12px",
                                    "background": "white",
                                    "borderRadius": "12px",
                                    "padding": "12px",
                                    "border": "1px solid #e6e6e6",
                                    "minWidth": 0,
                                },
                                children=[
                                    html.Div("τ Diagnostics", style={"fontWeight": 700, "marginBottom": "6px"}),
                                    html.Div(
                                        style={"display": "flex", "gap": "12px"},
                                        children=[
                                            html.Div(
                                                style={"flex": 1, "minWidth": 0},
                                                children=[
                                                    dcc.Graph(id="tau-fig", config={"displayModeBar": False}, style={"height": "320px"}),
                                                    html.Div(id="tau-help", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"}),
                                                ],
                                            ),
                                            html.Div(
                                                style={"flex": 1, "minWidth": 0},
                                                children=[
                                                    dcc.Graph(id="eval-test", config={"displayModeBar": False}, style={"height": "320px"}),
                                                    html.Div(id="eval-help", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"}),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
