# src/dashboards/boards/tau_sensitivity/layout.py
from __future__ import annotations

from typing import List

import pandas as pd
from dash import dcc, dash_table, html

from .queries import options_from_unique


# -----------------------------
# Shared styles
# -----------------------------
PAGE_BG = "#f6f7fb"
TEXT = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
TAB_BG = "#eef2f7"
TAB_ACTIVE = "white"

CARD = {
    "background": "white",
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
}

H2 = {
    "margin": "0",
    "fontSize": "22px",
    "fontWeight": 800,
    "letterSpacing": "-0.2px",
    "color": TEXT,
}

SUB = {
    "marginTop": "4px",
    "fontSize": "13px",
    "color": MUTED,
}

SECTION_TITLE = {
    "fontWeight": 800,
    "fontSize": "16px",
    "color": TEXT,
    "marginBottom": "6px",
}

SECTION_SUB = {
    "fontSize": "12px",
    "color": MUTED,
    "marginBottom": "12px",
}

TAB_STYLE = {
    "padding": "10px 18px",
    "fontWeight": 600,
    "fontSize": "13px",
    "color": "#64748b",
    "background": "#eef2f7",
    "border": "none",
    "borderRadius": "999px",
    "marginRight": "8px",
}

TAB_SELECTED_STYLE = {
    "padding": "10px 18px",
    "fontWeight": 700,
    "fontSize": "13px",
    "color": "#0f172a",
    "background": "white",
    "border": "1px solid #e2e8f0",
    "borderRadius": "999px",
    "marginRight": "8px",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.06)",
}
FIXED_SETTINGS = [
    {"param": "Execution Type", "value": "Close-to-Close"},
    {"param": "Backtesting Method", "value": "Rolling"},
    {"param": "Grid Size", "value": "100"},
    {"param": "Cutoff Time", "value": "4:00 PM"},
    {"param": "Realized Variance Intraday Frequency", "value": "5 Min"},
]


def section_card(title: str, children, subtitle: str | None = None) -> html.Div:
    child_list = children if isinstance(children, list) else [children]
    return html.Div(
        style={**CARD, "padding": "14px"},
        children=[
            html.Div(title, style=SECTION_TITLE),
            html.Div(subtitle, style=SECTION_SUB) if subtitle else None,
            *child_list,
        ],
    )


def kpi_card(title: str, value_id: str) -> html.Div:
    return html.Div(
        style={
            **CARD,
            "padding": "12px",
            "minWidth": 0,
            "minHeight": "78px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
        },
        children=[
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "color": MUTED,
                    "marginBottom": "6px",
                    "fontWeight": 500,
                },
            ),
            html.Div(
                id=value_id,
                style={
                    "fontSize": "22px",
                    "fontWeight": 800,
                    "color": TEXT,
                    "lineHeight": "1.1",
                },
            ),
        ],
    )


def styled_table(table_id: str, columns: list[dict], data=None):
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=data or [],
        fill_width=True,
        style_as_list_view=True,
        style_table={
            "overflowX": "auto",
            "borderRadius": "12px",
            "overflow": "hidden",
            "width": "100%",
        },
        style_header={
            "backgroundColor": "#f8fafc",
            "fontWeight": "700",
            "fontSize": "13px",
            "color": TEXT,
            "border": "none",
            "borderBottom": f"1px solid {BORDER}",
            "padding": "10px 12px",
        },
        style_cell={
            "backgroundColor": "white",
            "color": TEXT,
            "fontSize": "13px",
            "padding": "10px 12px",
            "border": "none",
            "borderBottom": "1px solid #f1f5f9",
            "textAlign": "left",
            "fontFamily": "Inter, system-ui, sans-serif",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fcfcfd"},
            {"if": {"column_id": "value"}, "textAlign": "right", "fontWeight": "600"},
        ],
    )


def filter_dropdown(label: str, component_id, options, value=None, placeholder=None, clearable=True):
    return html.Div(
        style={"marginBottom": "12px"},
        children=[
            html.Div(
                label,
                style={"fontSize": "12px", "color": MUTED, "marginBottom": "6px", "fontWeight": 600},
            ),
            dcc.Dropdown(
                id=component_id,
                options=options,
                value=value,
                placeholder=placeholder,
                clearable=clearable,
            ),
        ],
    )


def build_layout(
    *,
    runs_summary: pd.DataFrame,
    param_cols: List[str],
    default_opts: List[dict],
    default_value,
):
    return html.Div(
        style={
            "fontFamily": "Inter, system-ui, sans-serif",
            "padding": "16px",
            "background": PAGE_BG,
            "minHeight": "100vh",
        },
        children=[
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "420px minmax(0, 1fr)",
                    "gap": "16px",
                    "alignItems": "start",
                },
                children=[

                    # =========================
                    # LEFT: Sidebar
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
                            # html.Div("Filters", style=H2),
                            html.Div(
                                "Filter experiment runs and inspect strategy behavior.",
                                style={**SUB, "marginBottom": "16px"},
                            ),

                            html.Div("Filters", style={"fontWeight": 800, "marginBottom": "10px", "color": TEXT}),

                            *[
                                filter_dropdown(
                                    c,
                                    {"type": "param-dd", "name": c},
                                    options_from_unique(runs_summary[c]),
                                    value=None,
                                    placeholder=f"All {c}",
                                    clearable=True,
                                )
                                for c in param_cols
                            ],

                            html.Hr(style={"border": "none", "borderTop": f"1px solid {BORDER}", "margin": "14px 0"}),

                            filter_dropdown(
                                "Run",
                                "run-dd",
                                default_opts,
                                value=default_value,
                                placeholder="Select a run",
                                clearable=False,
                            ),

                            html.Div(
                                id="filtered-count",
                                style={"marginTop": "8px", "fontSize": "12px", "color": MUTED},
                            ),

                            html.Hr(style={"border": "none", "borderTop": f"1px solid {BORDER}", "margin": "16px 0"}),

                            section_card(
                                "Chosen Parameters",
                                styled_table(
                                    "chosen-params-table",
                                    [
                                        {"name": "Parameter", "id": "param"},
                                        {"name": "Value", "id": "value"},
                                    ],
                                ),
                            ),

                            html.Div(style={"height": "12px"}),

                            section_card(
                                "Fixed Settings",
                                styled_table(
                                    "fixed-settings-table",
                                    [
                                        {"name": "Setting", "id": "param"},
                                        {"name": "Value", "id": "value"},
                                    ],
                                    data=FIXED_SETTINGS,
                                ),
                            ),
                        ],
                    ),

                    # =========================
                    # RIGHT: Main content
                    # =========================
                    html.Div(
                        style={
                            "minWidth": 0,
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "12px",
                        },
                        children=[

                            html.Div(
                                children=[
                                    html.Div("Backtesting Results", style=H2),
                                    html.Div(
                                        "Inspect equity curves, portfolio metrics, and regime behavior for the selected run.",
                                        style={**SUB, "marginBottom": "12px"},
                                    ),
                                ]
                            ),

                            html.Div(
                                style={
                                    **CARD,
                                    "padding": "10px 10px 14px 10px",
                                },
                                children=[
                                    dcc.Tabs(
                                        id="main-tabs",
                                        value="tab-equity",
                                        parent_style={
                                            "border": "none",
                                            "marginBottom": "20px",
                                        },
                                        content_style={
                                            "border": "none",
                                            "padding": "0",
                                            "background": "transparent",
                                        },
                                        children=[

                                            # =========================
                                            # TAB 1: Equity
                                            # =========================
                                            dcc.Tab(
                                                label="Equity",
                                                value="tab-equity",
                                                style=TAB_STYLE,
                                                selected_style=TAB_SELECTED_STYLE,
                                                children=[
                                                    html.Div(
                                                        style={"padding": "12px", "minWidth": 0},
                                                        children=[
                                                            html.Div(
                                                                "XLE Regime State Strategy",
                                                                style=SECTION_TITLE,
                                                            ),
                                                            html.Div(
                                                                "Normalized equity, returns, weights, and state-threshold behavior.",
                                                                style=SECTION_SUB,
                                                            ),
                                                            dcc.Graph(
                                                                id="equity-fig",
                                                                config={
                                                                    "displayModeBar": False,
                                                                    "responsive": True,
                                                                },
                                                                style={"height": "74vh"},
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),

                                            # =========================
                                            # TAB 2: Statistics
                                            # =========================
                                            dcc.Tab(
                                                label="Statistics",
                                                value="tab-stats",
                                                style=TAB_STYLE,
                                                selected_style=TAB_SELECTED_STYLE,
                                                children=[
                                                    html.Div(
                                                        style={
                                                            "display": "grid",
                                                            "padding": "12px",
                                                            "gridTemplateColumns": "repeat(6, minmax(0, 1fr))",
                                                            "gap": "12px",
                                                            "marginBottom": "12px",
                                                        },
                                                        children=[
                                                            kpi_card("Sharpe", "kpi-sharpe"),
                                                            kpi_card("CAGR", "kpi-cagr"),
                                                            kpi_card("Max Drawdown", "kpi-mdd"),
                                                            kpi_card("Sharpe − B&H", "kpi-sharpe-diff"),
                                                            kpi_card("% Time Low Regime", "kpi-low-regime"),
                                                            kpi_card("Turnovers / Year", "kpi-turnover-yr"),
                                                        ],
                                                    ),

                                                    html.Div(
                                                        style={
                                                            "display": "grid",
                                                            "gridTemplateColumns": "1fr 1fr",
                                                            "gap": "12px",
                                                            "alignItems": "start",
                                                        },
                                                        children=[
                                                            section_card(
                                                                "Performance Summary",
                                                                styled_table(
                                                                    "performance-table",
                                                                    [
                                                                        {"name": "Metric", "id": "metric"},
                                                                        {"name": "Value", "id": "value"},
                                                                    ],
                                                                ),
                                                                subtitle="Core portfolio and benchmark comparison metrics.",
                                                            ),
                                                            section_card(
                                                                "Regime Behavior",
                                                                styled_table(
                                                                    "regime-table",
                                                                    [
                                                                        {"name": "Metric", "id": "metric"},
                                                                        {"name": "Value", "id": "value"},
                                                                    ],
                                                                ),
                                                                subtitle="Time in regime, turnover, and average regime durations.",
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
                    ),
                ],
            ),
        ],
    )

