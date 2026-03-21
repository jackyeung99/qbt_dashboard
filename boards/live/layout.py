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
SECTION_TITLE = {
    "fontWeight": 800,
    "fontSize": "16px",
    "marginBottom": "10px",
    "color": "#1f2937",
}

PLACEHOLDER_STYLE = {
    "height": "100%",
    "minHeight": "220px",
    "border": "1px dashed #d6d9e0",
    "borderRadius": "12px",
    "background": "#fbfcfe",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "textAlign": "center",
    "color": "#6b7280",
    "padding": "16px",
}

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

def info_card(title: str, markdown_text: str):
    return html.Div(
        style={**CARD, "padding": "12px"},
        children=[
            html.Div(title, style=SECTION_TITLE),
            dcc.Markdown(
                markdown_text,
                mathjax=True,
                style={
                    "fontSize": "14px",
                    "lineHeight": 1.6,
                    "color": "#4b5563",
                },
            ),
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
                    # **CARD,
                    "padding": "14px",
                    "marginBottom": "12px",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "gap": "12px",
                },
                children=[
                    html.Div(
                        children=[
                            html.Div(title, style=H2),
                            html.Div(
                                subtitle or "Live monitoring for a multi-ETF volatility regime portfolio.",
                                style=SUB,
                            ),
                        ]
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Div(
                                "Live",
                                id="live_status_badge",
                                style={
                                    "fontSize": "13px",
                                    "color": "#0f5132",
                                    "background": "#d1e7dd",
                                    "padding": "6px 10px",
                                    "borderRadius": "999px",
                                    "fontWeight": 700,
                                    "whiteSpace": "nowrap",
                                },
                            ),
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
                ],
            ),

            # Top summary row
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "2fr 1.2fr",
                    "gap": "12px",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        style={**CARD, "padding": "12px"},
                        children=[
                            html.Div("Portfolio Performance", style=SECTION_TITLE),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(6, minmax(0, 1fr))",
                                    "gap": "12px",
                                },
                                children=[
                                    kpi_card("Total PNL", "kpi-total-pnl"),
                                    kpi_card("Sharpe", "kpi-sharpe"),
                                    kpi_card("CAGR", "kpi-cagr"),
                                    kpi_card("Max Drawdown", "kpi-mdd"),
                                    kpi_card("Volatility", "kpi-return-vol"),
                                    kpi_card("Avg. Daily Return", "kpi-mean-return"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={**CARD, "padding": "12px"},
                        children=[
                            html.Div("Current Positioning", style=SECTION_TITLE),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                    "gap": "12px",
                                },
                                children=[
                                    kpi_card("Active ETFs", "kpi-active-etfs"),
                                    kpi_card("Gross Exposure", "kpi-gross-exposure"),
                                    kpi_card("Net Exposure", "kpi-net-exposure"),
                                    kpi_card("Cash", "kpi-cash"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Strategy explanation
            html.Div(
                style={
                    "display": "grid",
                    # "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                    "gap": "12px",
                    "marginBottom": "12px",
                },
                children=[

                    html.Div(
                    style={**CARD, "padding": "12px"},
                    children=[
            
                        html.Div("Volatility Regime Strategy", style=SECTION_TITLE),
                        dcc.Markdown(
                            ''' 
                                the core hypothesis of this strategy is that the peformance of etfs can be optimized based of the underlying volatility. 
                                
                                We consider the intraday realized volatility at day t as the proxy for volatility of the underlying etf. We use 5 min intraday bars  

                                ### Volatility Proxy
                                ### Regime Classification 
                                ### Optimal Threshold $\\tau^*$
                                ### Combining Multiple ETFs 

                            ''',
                            mathjax=True,
                            style={
                                "fontSize": "18px",
                                "lineHeight": 1.6,
                                "color": "#4b5563",
                            },
                            ),
                    ])
                ],
            ),

            # Portfolio overview section
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "2fr 1fr",
                    "gap": "12px",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        style={**CARD, "padding": "12px", "minHeight": "420px"},
                        children=[
                            html.Div("Portfolio Overview", style=SECTION_TITLE),
                            html.Div(
                                id="equity-fig",
                                style=PLACEHOLDER_STYLE,
                                children="Reserve space for total portfolio equity / drawdown / performance view",
                            ),
                        ],
                    ),
                    html.Div(
                        style={**CARD, "padding": "12px", "minHeight": "420px"},
                        children=[
                            html.Div("Allocation & Contribution", style=SECTION_TITLE),
                            html.Div(
                                id="allocation_placeholder",
                                style=PLACEHOLDER_STYLE,
                                children="Reserve space for current allocation, sleeve contribution, or weight summary",
                            ),
                        ],
                    ),
                ],
            ),

            # ETF drilldown
            html.Div(
                style={**CARD, "padding": "12px", "marginBottom": "12px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "12px",
                            "gap": "12px",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Div("ETF Drilldown", style=SECTION_TITLE),
                            dcc.Dropdown(
                                id="etf-selector",
                                placeholder="Select ETF",
                                style={"minWidth": "240px"},
                            ),
                        ],
                    ),

                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                            "gap": "12px",
                            "marginBottom": "12px",
                        },
                        children=[
                            kpi_card("ETF Return", "etf-kpi-return"),
                            kpi_card("ETF Sharpe", "etf-kpi-sharpe"),
                            kpi_card("Current Regime", "etf-kpi-regime"),
                            kpi_card("Target Weight", "etf-kpi-weight"),
                        ],
                    ),

                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "2fr 1fr",
                            "gap": "12px",
                        },
                        children=[
                            html.Div(
                                style={"minHeight": "360px"},
                                children=[
                                    html.Div(
                                        id="etf_main_placeholder",
                                        style=PLACEHOLDER_STYLE,
                                        children="Reserve space for ETF sleeve equity / price overlay / detail view",
                                    )
                                ],
                            ),
                            html.Div(
                                style={"minHeight": "360px"},
                                children=[
                                    html.Div(
                                        id="etf_side_placeholder",
                                        style=PLACEHOLDER_STYLE,
                                        children="Reserve space for regime, state variable, and weight history",
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),


            # ),
        ],
    )