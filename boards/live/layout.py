from __future__ import annotations

from dash import dcc, html


THEME = {
    "colors": {
        "page_bg": "#f6f7fb",
        "surface": "#ffffff",
        "surface_alt": "#fbfcfe",
        "border": "#e6e6e6",
        "border_soft": "#d6d9e0",
        "text": "#111827",
        "text_muted": "#6b7280",
        "text_soft": "#4b5563",
        "heading": "#1f2937",
        "badge_live_bg": "#d1e7dd",
        "badge_live_text": "#0f5132",
        "badge_neutral_bg": "#f2f3f7",
        "shadow": "0 1px 2px rgba(0,0,0,0.04)",
    },
    "font": {
        "family": "system-ui",
        "title": "32px",
        "subtitle": "16px",
        "section": "16px",
        "kpi_label": "12px",
        "kpi_value": "20px",
        "badge": "18px",
        "body": "18px",
    },
    "space": {
        "page_pad": "16px",
        "container_pad": "16px",
        "section_gap": "12px",
        "card_pad": "12px",
        "panel_gap": "10px",
        "title_gap": "4px",
        "kpi_gap": "12px",
        "kpi_label_gap": "6px",
        "header_y": "14px 0",
        "placeholder_pad": "16px",
    },
    "radius": {
        "card": "14px",
        "inner": "12px",
        "pill": "999px",
    },
    "layout": {
        "max_width": "1900px",
        "kpi_row_cols": "repeat(6, minmax(0, 1fr))",
        "positioning_cols": "repeat(2, minmax(0, 1fr))",
        "top_grid_cols": "2fr 1.2fr",
        "overview_grid_cols": "2fr 1fr",
        "etf_kpi_cols": "repeat(4, minmax(0, 1fr))",
        "etf_detail_cols": "2fr 1fr",
        "overview_min_height": "420px",
        "detail_min_height": "360px",
        "dropdown_min_width": "240px",
    },
}


def sx_card() -> dict:
    c = THEME["colors"]
    r = THEME["radius"]
    return {
        "background": c["surface"],
        "border": f'1px solid {c["border"]}',
        "borderRadius": r["card"],
        "boxShadow": c["shadow"],
        "boxSizing": "border-box",
    }


def sx_panel() -> dict:
    s = THEME["space"]
    return {
        **sx_card(),
        "padding": s["card_pad"],
        "display": "flex",
        "flexDirection": "column",
        "gap": s["panel_gap"],
        "overflow": "hidden",
    }


def sx_section_title() -> dict:
    c = THEME["colors"]
    f = THEME["font"]
    return {
        "fontWeight": 800,
        "fontSize": f["section"],
        "marginBottom": THEME["space"]["panel_gap"],
        "color": c["heading"],
    }


def sx_placeholder() -> dict:
    c = THEME["colors"]
    r = THEME["radius"]
    s = THEME["space"]
    return {
        "width": "100%",
        "flex": "1",
        "minHeight": 0,
        "border": f'1px dashed {c["border_soft"]}',
        "borderRadius": r["inner"],
        "background": c["surface_alt"],
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "textAlign": "center",
        "color": c["text_muted"],
        "padding": s["placeholder_pad"],
        "boxSizing": "border-box",
        "overflow": "hidden",
    }


def page_shell(children) -> html.Div:
    return html.Div(
        style={
            "fontFamily": THEME["font"]["family"],
            "background": THEME["colors"]["page_bg"],
            "minHeight": "100vh",
            "width": "100%",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                style={
                    "maxWidth": THEME["layout"]["max_width"],
                    "width": "100%",
                    "margin": "0 auto",
                    "padding": THEME["space"]["container_pad"],
                    "boxSizing": "border-box",
                },
                children=children,
            )
        ],
    )


def kpi_card(title: str, value_id: str) -> html.Div:
    c = THEME["colors"]
    f = THEME["font"]
    s = THEME["space"]

    return html.Div(
        style={
            **sx_card(),
            "padding": s["card_pad"],
            "minWidth": 0,
            "minHeight": "74px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
        },
        children=[
            html.Div(
                title,
                style={
                    "fontSize": f["kpi_label"],
                    "color": c["text_muted"],
                    "marginBottom": s["kpi_label_gap"],
                },
            ),
            html.Div(
                id=value_id,
                style={
                    "fontSize": f["kpi_value"],
                    "fontWeight": 800,
                    "color": c["text"],
                },
            ),
        ],
    )


def info_badge(text: str | None = None, *, id: str | None = None, live: bool = False) -> html.Div:
    c = THEME["colors"]
    f = THEME["font"]

    bg = c["badge_live_bg"] if live else c["badge_neutral_bg"]
    fg = c["badge_live_text"] if live else c["text_soft"]
    border = "none" if live else f'1px solid {c["border"]}'

    return html.Div(
        text,
        id=id,
        style={
            "fontSize": f["badge"],
            "color": fg,
            "background": bg,
            "padding": "6px 10px",
            "borderRadius": THEME["radius"]["pill"],
            "fontWeight": 700 if live else 600,
            "whiteSpace": "nowrap",
            "border": border,
        },
    )


def section_card(title: str, children, *, style: dict | None = None) -> html.Div:
    return html.Div(
        style={**sx_card(), "padding": THEME["space"]["card_pad"], **(style or {})},
        children=[
            html.Div(title, style=sx_section_title()),
            *children,
        ],
    )


def panel_card(title: str, body_id: str, body_text: str, *, min_height: str) -> html.Div:
    return html.Div(
        style={**sx_panel(), "minHeight": min_height},
        children=[
            html.Div(title, style=sx_section_title()),
            html.Div(id=body_id, style=sx_placeholder(), children=body_text),
        ],
    )


def build_layout(*, title: str = "Live Portfolio", subtitle: str | None = None):
    c = THEME["colors"]
    f = THEME["font"]
    s = THEME["space"]
    l = THEME["layout"]

    return page_shell(
        [
            dcc.Location(id="url"),

            html.Div(
                style={
                    "padding": s["header_y"],
                    "marginBottom": s["section_gap"],
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "gap": s["section_gap"],
                },
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                title,
                                style={
                                    "margin": "0",
                                    "fontSize": f["title"],
                                    "fontWeight": 800,
                                    "letterSpacing": "-0.2px",
                                    "color": c["text"],
                                },
                            ),
                            html.Div(
                                subtitle or "Live monitoring for a multi-ETF volatility regime portfolio.",
                                style={
                                    "marginTop": s["title_gap"],
                                    "fontSize": f["subtitle"],
                                    "color": c["text_muted"],
                                },
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
                            info_badge("Live", id="live_status_badge", live=True),
                            info_badge(id="generated_at"),
                        ],
                    ),
                ],
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": l["top_grid_cols"],
                    "gap": s["section_gap"],
                    "marginBottom": s["section_gap"],
                },
                children=[
                    section_card(
                        "Portfolio Performance",
                        [
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": l["kpi_row_cols"],
                                    "gap": s["kpi_gap"],
                                },
                                children=[
                                    kpi_card("Total PNL", "kpi-total-pnl"),
                                    kpi_card("Sharpe", "kpi-sharpe"),
                                    kpi_card("CAGR", "kpi-cagr"),
                                    kpi_card("Max Drawdown", "kpi-mdd"),
                                    kpi_card("Volatility", "kpi-return-vol"),
                                    kpi_card("Avg. Daily Return", "kpi-mean-return"),
                                ],
                            )
                        ],
                    ),
                    section_card(
                        "Current Positioning",
                        [
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": l["positioning_cols"],
                                    "gap": s["kpi_gap"],
                                },
                                children=[
                                    kpi_card("Active ETFs", "kpi-active-etfs"),
                                    kpi_card("Gross Exposure", "kpi-gross-exposure"),
                                    kpi_card("Net Exposure", "kpi-net-exposure"),
                                    kpi_card("Cash", "kpi-cash"),
                                ],
                            )
                        ],
                    ),
                ],
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gap": s["section_gap"],
                    "marginBottom": s["section_gap"],
                },
                children=[
                    section_card(
                        "Volatility Regime Strategy",
                        [
                            dcc.Markdown(
                                """
The core hypothesis of this strategy is that ETF performance can be improved by conditioning on the underlying volatility regime.

We use intraday realized volatility at day $t$ as the volatility proxy for the ETF, computed from 5-minute intraday bars.

### Volatility Proxy
### Regime Classification
### Optimal Threshold $\\tau^*$
### Combining Multiple ETFs
                                """,
                                mathjax=True,
                                style={
                                    "fontSize": f["body"],
                                    "lineHeight": 1.6,
                                    "color": c["text_soft"],
                                },
                            )
                        ],
                    )
                ],
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": l["overview_grid_cols"],
                    "gap": s["section_gap"],
                    "marginBottom": s["section_gap"],
                },
                children=[
                    panel_card(
                        "Portfolio Overview",
                        "equity-fig",
                        "Reserve space for total portfolio equity / drawdown / performance view",
                        min_height=l["overview_min_height"],
                    ),
                    panel_card(
                        "Allocation & Contribution",
                        "allocation_placeholder",
                        "Reserve space for current allocation, sleeve contribution, or weight summary",
                        min_height=l["overview_min_height"],
                    ),
                ],
            ),

            section_card(
                "ETF Drilldown",
                [
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": s["section_gap"],
                            "gap": s["section_gap"],
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Div("", style={"display": "none"}),
                            dcc.Dropdown(
                                id="etf-selector",
                                placeholder="Select ETF",
                                style={"minWidth": l["dropdown_min_width"]},
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": l["etf_kpi_cols"],
                            "gap": s["kpi_gap"],
                            "marginBottom": s["section_gap"],
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
                            "gridTemplateColumns": l["etf_detail_cols"],
                            "gap": s["section_gap"],
                        },
                        children=[
                            html.Div(
                                style={**sx_panel(), "minHeight": l["detail_min_height"]},
                                children=[
                                    html.Div(
                                        id="etf_main_placeholder",
                                        style=sx_placeholder(),
                                        children="Reserve space for ETF sleeve equity / price overlay / detail view",
                                    )
                                ],
                            ),
                            html.Div(
                                style={**sx_panel(), "minHeight": l["detail_min_height"]},
                                children=[
                                    html.Div(
                                        id="etf_side_placeholder",
                                        style=sx_placeholder(),
                                        children="Reserve space for regime, state variable, and weight history",
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
                style={"marginBottom": s["section_gap"]},
            ),
        ]
    )