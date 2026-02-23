# src/dashboards/boards/tau_sensitivity/layout.py
from __future__ import annotations

from dash import dcc, html


# -----------------------------
# Shared styles
# -----------------------------
CARD = {
    "background": "white",
    "border": "1px solid #e6e6e6",
    "borderRadius": "14px",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
}

H2 = {"margin": "0", "fontSize": "30px", "fontWeight": 800, "letterSpacing": "-0.2px"}
SUB = {"marginTop": "4px", "fontSize": "16px", "color": "#666"}


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


def build_layout(*, title: str = "Live Portfolio", subtitle: str | None = None):
    return html.Div(
        style={
            "fontFamily": "system-ui",
            "padding": "16px",
            "background": "#f6f7fb",
            "minHeight": "100vh",
        },
        children=[
            dcc.Location(id="url"), 
            # Header
            html.Div(
                style={
                    **CARD,
                    "padding": "14px",
                    "marginBottom": "12px",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "gap": "12px",
                },
                children=[

                    # Left: title + subtitle
                    html.Div(
                        children=[
                            html.Div(title, style=H2),
                            html.Div(
                                subtitle or "KPIs + equity curve from the latest portfolio snapshot.",
                                style=SUB,
                            ),
                        ]
                    ),

                    # Right: generated time badge
                    html.Div(
                        id="generated_at",
                        style={
                            "fontSize": "14px",
                            "color": "#555",
                            "background": "#f2f3f7",
                            "padding": "6px 10px",
                            "borderRadius": "999px",
                            "whiteSpace": "nowrap",
                            "border": "1px solid #e6e6e6",
                            "fontWeight": 600,
                        },
                    ),
                ],
            ),

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
                    kpi_card("Average Daily Return", "kpi-mean-return"),
                    kpi_card("Return Volatility", "kpi-return-vol"),
                ],
            ),

            # Equity chart
            html.Div(
                style={**CARD, "padding": "12px", "minWidth": 0},
                children=[
                    html.Div("Equity Curve", style={"fontWeight": 800, "marginBottom": "8px"}),
                    dcc.Graph(
                        id="equity-fig",
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "72vh"},
                    ),
                ],
            ),
        ],
    )
