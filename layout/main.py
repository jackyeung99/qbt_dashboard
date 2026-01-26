# dashboard/layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html

from .components import card, kpi, spacer, nice_table


def make_layout(max_width_px: int = 1500):
    # LEFT: Filters + Runs
    controls = card(
        "Filters",
        [
            dbc.Alert(id="status", color="secondary", className="py-2", style={"marginBottom": "12px"}),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Strategy", className="mb-1"),
                            dcc.Dropdown(id="strategy_dd", placeholder="All strategies"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Universe", className="mb-1"),
                            dcc.Dropdown(id="universe_dd", placeholder="All universes"),
                        ],
                        md=6,
                    ),
                ],
                className="g-2",
            ),
            spacer(10),

            # ✅ Sort controls (not shown in table)
            dbc.Label("Sort by", className="mb-1"),
            dcc.Dropdown(
                id="sort_metric_dd",
                options=[
                    {"label": "Created time", "value": "created_at_utc"},
                    {"label": "Sharpe", "value": "sharpe"},
                    {"label": "CAGR", "value": "cagr"},
                    {"label": "Max Drawdown", "value": "max_dd"},
                    {"label": "Volatility", "value": "volatility"},
                ],
                value="created_at_utc",
                clearable=False,
            ),
            spacer(6),

            dbc.Label("Direction", className="mb-1"),
            dbc.RadioItems(
                id="sort_dir",
                options=[
                    {"label": "Desc", "value": "desc"},
                    {"label": "Asc", "value": "asc"},
                ],
                value="desc",
                inline=True,
                className="mb-2",
            ),
            spacer(10),

            dbc.Label("Run", className="mb-1"),
            dcc.Dropdown(id="run_dd", placeholder="Select a run..."),

            spacer(6),

            dbc.Label("Compare with", className="mb-1"),
            dcc.Dropdown(
                id="compare_run_dd",
                placeholder="Select run to compare...",
                clearable=True,
            ),
        ],
    )

    runs_list = card(
        "Runs",
        [
            nice_table(
                "runs_table",
                page_size=12,
                height_px=540,
                nowrap=True,
                selectable=True,
                filter_action="none",   # cleaner
                sort_action="none",   # click headers to sort too (optional)
                sort_mode="multi",
                
            )
        ],
    )

    left = dbc.Stack([controls, runs_list], gap=3)

    # RIGHT: Selected run (badges + kv) + KPIs + chart
    selected_run = dbc.Card(
        [
            dbc.CardHeader("Selected run", style={"fontWeight": 600, "fontSize": "0.95rem"}),
            dbc.CardBody(
                [
                    html.Div(id="run_badges"),
                    spacer(8),
                    html.Div("Model parameters", style={"fontWeight": 600, "opacity": 0.7, "fontSize": "0.85rem"}),
                    spacer(6),
                    html.Div(id="params_kv"),
                ],
                style={"padding": "14px"},
            ),
        ],
        className="shadow-sm",
    )

    kpis = dbc.Row(
        [
            dbc.Col(kpi("Sharpe", "kpi_sharpe"), md=3),
            dbc.Col(kpi("CAGR", "kpi_cagr", "Annualized"), md=3),
            dbc.Col(kpi("Max Drawdown", "kpi_mdd"), md=3),
            dbc.Col(kpi("Volatility", "kpi_vol", "Annualized"), md=3),
        ],
        className="g-3",
    )

    equity = card(
        "Equity curve",
        dcc.Graph(
            id="equity_fig",
            style={"height": "520px"},
            config={"displayModeBar": False, "displaylogo": False, "responsive": True},
        ),
    )

    right = dbc.Stack([selected_run, kpis, equity], gap=3)

    # Page
    return dbc.Container(
        fluid=True,
        style={
            "maxWidth": f"{max_width_px}px",
            "paddingTop": "18px",
            "paddingBottom": "38px",
            "backgroundColor": "#BBBBBB",
            "minHeight": "100vh",
        },
        children=[
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H2("Backtesting Dashboard", style={"marginBottom": "4px"}),
                            html.Div(
                                "Browse strategies and runs, inspect parameters, and compare performance.",
                                style={"opacity": 0.7},
                            ),
                        ]
                    )
                )
            ),
            html.Hr(style={"marginTop": "14px", "marginBottom": "18px"}),

            dbc.Row(
                [
                    dbc.Col(left, lg=4, md=5, sm=12),
                    dbc.Col(right, lg=8, md=7, sm=12),
                ],
                className="g-4",
            ),
        ],
    )
