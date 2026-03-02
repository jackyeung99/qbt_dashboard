# src/dashboards/boards/tau_sensitivity/layout.py
from __future__ import annotations

from typing import List

import pandas as pd
from dash import dcc, dash_table, html

from .queries import options_from_unique


# -----------------------------
# Shared styles
# -----------------------------
# PAGE_BG = "#f6f7fb"

CARD = {
    "background": "white",
    "border": "1px solid #e6e6e6",
    "borderRadius": "14px",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
}

H2 = {"margin": "0", "fontSize": "22px", "fontWeight": 800, "letterSpacing": "-0.2px"}
SUB = {"marginTop": "4px", "fontSize": "13px", "color": "#666"}

FIXED_SETTINGS = [
        {"param": "Execution Type", "value": "Close-to-Close"},
        {"param": "Backtesting Method", "value": "Rolling"},
        {"param": "Grid Size", "value": "100"},
        {"param": "Cutoff Time", "value": "4:00 PM"},
        {"param": "Realized Variance Intraday Frequency", "value": "5 Min"},
    ]

def kpi_card(title: str, value_id: str) -> html.Div:
    return html.Div(
        style={
            **CARD,
            "padding": "12px 12px",
            "minWidth": 0,
            "minHeight": "74px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"}),
            html.Div(id=value_id, style={"fontSize": "20px", "fontWeight": 800}),
        ],
    )


def build_layout(*, runs_summary: pd.DataFrame, param_cols: List[str], default_opts: List[dict], default_value):
    return html.Div(
    style={
        "fontFamily": "system-ui", "padding": "16px", "background": "#f6f7fb",
        "display": "grid",
        "gridTemplateColumns": "520px minmax(0, 1fr)",
        "gap": "14px",
        "alignItems": "start",
    },
    children=[

        # =========================
        # LEFT: Filters + Run Stats
        # =========================
        html.Div(
            style={
                **CARD,
                "padding": "14px",
                "position": "sticky",
                "top": "12px",
                "maxHeight": "calc(100vh - 24px)",
                "overflowY": "auto",
            },
            children=[
                html.Div("Filters", style={"fontWeight": 800, "marginBottom": "10px"}),

                *[
                    html.Div(
                        style={"marginBottom": "10px"},
                        children=[
                            html.Div(
                                c,
                                style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"},
                            ),
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

                html.Hr(style={"border": "none", "borderTop": "1px solid #eee", "margin": "12px 0"}),

                html.Div("Run", style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="run-dd",
                    options=default_opts,
                    value=default_value,
                    placeholder="Select a run",
                    clearable=False,
                ),

                html.Div(
                    id="filtered-count",
                    style={"marginTop": "10px", "fontSize": "12px", "color": "#666"},
                ),

                html.Hr(style={"border": "none", "borderTop": "1px solid #eee", "margin": "16px 0"}),

                                # -------------------------
                # Run Configuration
                # -------------------------
                html.Div("Run Configuration", style={"fontWeight": 800, "marginBottom": "8px"}),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "10px"},
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    "Chosen Parameters",
                                    style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"},
                                ),
                                dash_table.DataTable(
                                    id="chosen-params-table",
                                    columns=[{"name": "Parameter", "id": "param"}, {"name": "Value", "id": "value"}],
                                    data=[],
                                    style_table={
                                        "border": "1px solid #eee",
                                        "borderRadius": "10px",
                                        "overflow": "hidden",
                                    },
                                    style_cell={"fontSize": 12, "padding": "8px", "border": "none"},
                                    style_header={"fontWeight": 800, "background": "#fafafa", "borderBottom": "1px solid #eee"},
                                ),
                            ]
                        ),

                        dash_table.DataTable(
                            id="fixed-settings-table",
                            columns=[{"name": "Setting", "id": "param"}, {"name": "Value", "id": "value"}],
                            data=FIXED_SETTINGS,
                            style_table={
                                "border": "1px solid #eee",
                                "borderRadius": "10px",
                                "overflow": "hidden",
                                "width": "100%",          # 👈 important in narrow columns
                            },
                            style_cell={
                                "fontSize": 12,
                                "padding": "8px",
                                "border": "none",
                                "whiteSpace": "normal",   # 👈 avoids truncation
                                "height": "auto",
                            },
                            style_header={"fontWeight": 800, "background": "#fafafa", "borderBottom": "1px solid #eee"},
                        )
                    ],
                ),

                html.Hr(style={"border": "none", "borderTop": "1px solid #eee", "margin": "16px 0"}),
                # -------------------------
                # Run Stats (moved here)
                # -------------------------
                html.Div("Run Stats", style={"fontWeight": 800, "marginBottom": "8px"}),

                dash_table.DataTable(
                    id="stats-table",
                    columns=[
                        {"name": "metric", "id": "metric"},
                        {"name": "value", "id": "value"},
                    ],
                    data=[],
                    style_table={
                        "height": "360px",
                        "overflowY": "auto",
                        "border": "1px solid #eee",
                        "borderRadius": "10px",
                    },
                    style_cell={
                        "fontSize": 12,
                        "padding": "8px",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "border": "none",
                    },
                    style_header={
                        "fontWeight": 800,
                        "background": "#fafafa",
                        "borderBottom": "1px solid #eee",
                    },
                ),
            ],
        ),

        # =========================
        # RIGHT: Content
        # =========================
        html.Div(
            style={"minWidth": 0},
            children=[

                # KPI Row
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(5, minmax(0, 1fr))",
                        "gap": "12px",
                        "marginBottom": "12px",
                    },
                    children=[
                        kpi_card("Sharpe", "kpi-sharpe"),
                        kpi_card("CAGR", "kpi-cagr"),
                        kpi_card("Max Drawdown", "kpi-mdd"),
                        kpi_card("Turnover", "kpi-turnover"),
                        kpi_card("τ*", "kpi-tau"),
                    ],
                ),

                # Full-width equity
                html.Div(
                    style={**CARD, "padding": "12px", "minWidth": 0},
                    children=[
                        html.Div(
                            "XLE Regime State Strategy",
                            style={"fontWeight": 800, "marginBottom": "8px"},
                        ),
                        dcc.Graph(
                            id="equity-fig",
                            config={"displayModeBar": False, "responsive": True},
                            # style={"height": "920px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)