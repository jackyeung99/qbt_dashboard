import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

def build_pretty_label(row: pd.Series) -> str:
    parts = [
        f"{row.get('intraday_freq','?')}",
        f"cutoff={row.get('cutoff','?')}",
        f"yrs={row.get('selection_years','?')}",
        f"grid={row.get('grid_size','?')}",
        f"rf={row.get('rf','?')}",
        f"tc={row.get('transaction_cost','?')}bps",
        f"τ*={row.get('tau_star', float('nan')):.3g}" if pd.notna(row.get("tau_star", None)) else "τ*=?",
        f"Sh(c2c)={row.get('c2c_sharpe', float('nan')):.2f}" if pd.notna(row.get("c2c_sharpe", None)) else "Sh=?",
    ]
    return " | ".join(parts)

def options_from_unique(s: pd.Series):
    vals = sorted([v for v in s.dropna().unique().tolist()])
    return [{"label": str(v), "value": v} for v in vals]

def kpi_card(title: str, value_id: str):
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

def make_app(runs_summary: pd.DataFrame, equity_curves: pd.DataFrame, thresholds: pd.DataFrame | None):
    rs = runs_summary.copy()

    # ---- checks ----
    if "run_id" not in rs.columns:
        raise ValueError("runs_summary must contain 'run_id'")
    for c in ["run_id", "strategy", "timestamp", "equity"]:
        if c not in equity_curves.columns:
            raise ValueError(f"equity_curves must contain '{c}'")

    # ---- preprocess once (server-side) ----
    ec = equity_curves.copy()
    ec["timestamp"] = pd.to_datetime(ec["timestamp"])

    # (run_id, strategy) -> DF
    EC_MAP = {
        (rid, strat): g.sort_values("timestamp")
        for (rid, strat), g in ec.groupby(["run_id", "strategy"], sort=False)
    }

    TH_MAP = None
    if thresholds is not None:
        th = thresholds.copy()
        if "run_id" not in th.columns:
            raise ValueError("thresholds must contain 'run_id'")
        TH_MAP = {rid: g for rid, g in th.groupby("run_id", sort=False)}

    # labels
    if "run_label" not in rs.columns:
        rs["run_label"] = rs.apply(build_pretty_label, axis=1)

    # params for filters
    param_cols = [
        "intraday_freq",
        "cutoff",
        "selection_years",
        "grid_size",
        "rf",
        "transaction_cost",
        "tau_quantile_bounds",
    ]
    param_cols = [c for c in param_cols if c in rs.columns]

    # IMPORTANT: don't ship huge option lists on page load
    # start with a small default subset
    rs_sorted = rs.sort_values("run_label")
    rs_default = rs_sorted.head(200)  # cap for initial payload
    default_opts = [{"label": r["run_label"], "value": r["run_id"]} for r in rs_default.to_dict("records")]
    default_value = default_opts[0]["value"] if default_opts else None

    app = Dash(__name__)
    app.layout = html.Div(
    style={"fontFamily": "system-ui", "padding": "16px", "background": "#f6f7fb"},
    children=[
        html.H2("Sweep Dashboard", style={"margin": "0 0 10px 0"}),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "360px 1fr",
                "gap": "12px",
                "minWidth": 0,          # ✅ allow grid children to shrink
            },
            children=[
                # LEFT
                html.Div(
                    style={
                        "background": "white",
                        "borderRadius": "12px",
                        "padding": "12px",
                        "border": "1px solid #e6e6e6",
                    },
                    children=[
                        html.Div("Filters", style={"fontWeight": 700, "marginBottom": "8px"}),

                        *[
                            html.Div(
                                style={"marginBottom": "10px"},
                                children=[
                                    html.Div(c, style={"fontSize": "12px", "color": "#666"}),
                                    dcc.Dropdown(
                                        id={"type": "param-dd", "name": c},
                                        options=options_from_unique(rs[c]),
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

                        html.Div(
                            id="filtered-count",
                            style={"marginTop": "10px", "fontSize": "12px", "color": "#666"},
                        ),
                    ],
                ),

                # RIGHT
                html.Div(
                    style={"minWidth": 0},   # ✅ critical: allow right panel to shrink
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "10px",
                                "flexWrap": "wrap",
                                "marginBottom": "12px",
                                "minWidth": 0,  # ✅
                            },
                            children=[
                                kpi_card("Sharpe", "kpi-sharpe"),
                                kpi_card("CAGR", "kpi-cagr"),
                                kpi_card("Max Drawdown", "kpi-mdd"),
                                kpi_card("Turnover", "kpi-turnover"),
                                kpi_card("τ*", "kpi-tau"),
                            ],
                        ),

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 420px",
                                "gap": "12px",
                                "minWidth": 0,     # ✅ allow inner grid to shrink
                            },
                            children=[
                                html.Div(
                                    style={
                                        "background": "white",
                                        "borderRadius": "12px",
                                        "padding": "12px",
                                        "border": "1px solid #e6e6e6",
                                        "minWidth": 0,  # ✅ critical: this card must be shrinkable
                                    },
                                    children=[
                                        html.Div("Equity Curve", style={"fontWeight": 700, "marginBottom": "6px"}),
                                        dcc.Graph(
                                            id="equity-fig",
                                            config={"displayModeBar": False, "responsive": True},
                                            style={"width": "100%", "minWidth": 0, "height": "420px"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "background": "white",
                                        "borderRadius": "12px",
                                        "padding": "12px",
                                        "border": "1px solid #e6e6e6",
                                        "minWidth": 0,  # ✅
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
                                "minWidth": 0,  # ✅
                            },
                            children=[
                                html.Div("τ Diagnostics", style={"fontWeight": 700, "marginBottom": "6px"}),
                                dcc.Graph(
                                    id="tau-fig",
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"width": "100%", "minWidth": 0, "height": "320px"},
                                ),
                                html.Div(id="tau-help", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


    def filter_runs(filters: dict) -> pd.DataFrame:
        out = rs
        for k, v in filters.items():
            if v is None:
                continue
            out = out[out[k] == v]
        return out

    # Build Inputs list dynamically (no dummy Inputs)
    dropdown_inputs = [Input({"type": "param-dd", "name": c}, "value") for c in param_cols]

    @callback(
        Output("run-dd", "options"),
        Output("run-dd", "value"),
        Output("filtered-count", "children"),
        dropdown_inputs,
        prevent_initial_call=True,   # key: do NOT fire on initial page load
    )
    def update_run_dropdown(*vals):
        filters = {c: v for c, v in zip(param_cols, vals)}
        dff = filter_runs(filters).sort_values("run_label")

        # cap dropdown size so it stays fast
        dff = dff.head(500)

        opts = [{"label": r["run_label"], "value": r["run_id"]} for r in dff.to_dict("records")]
        default_value = opts[0]["value"] if opts else None
        return opts, default_value, f"{len(dff)} run(s) in list (capped to 500)."
    
    def plot_equity_with_bh(run_id: str, strategy: str):
        fig = go.Figure()

        # ---- Always try to plot BH first ----
        bh = EC_MAP.get((run_id, "bh"))
        if bh is not None and not bh.empty:
            fig.add_trace(
                go.Scatter(
                    x=bh["timestamp"],
                    y=bh["equity"],
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="gray", width=2, dash="dash"),
                )
            )

        # ---- Overlay selected strategy (unless it's bh itself) ----
        if strategy != "bh":
            strat = EC_MAP.get((run_id, strategy))
            if strat is not None and not strat.empty:
                fig.add_trace(
                    go.Scatter(
                        x=strat["timestamp"],
                        y=strat["equity"],
                        mode="lines",
                        name=strategy.upper(),
                        line=dict(width=3),
                    )
                )

        # ---- Fallback if nothing exists ----
        if len(fig.data) == 0:
            fig.add_annotation(
                text="No equity data for this run",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14),
            )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title="Equity",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    @callback(
        Output("equity-fig", "figure"),
        Output("kpi-sharpe", "children"),
        Output("kpi-cagr", "children"),
        Output("kpi-mdd", "children"),
        Output("kpi-turnover", "children"),
        Output("kpi-tau", "children"),
        Output("stats-table", "data"),
        Output("tau-fig", "figure"),
        Output("tau-help", "children"),
        Input("run-dd", "value"),
        Input("strategy-radio", "value"),
        prevent_initial_call=False,)
    def render_run(run_id, strategy):
        empty_fig = px.line(pd.DataFrame({"x": [], "y": []}), x="x", y="y")

        if run_id is None:
            return empty_fig, "-", "-", "-", "-", "-", [], empty_fig, "Select a run."

        row_df = rs.loc[rs["run_id"] == run_id]
        if row_df.empty:
            return empty_fig, "-", "-", "-", "-", "-", [], empty_fig, "Run not found."
        row = row_df.iloc[0]

        # equity (O(1))
        eq_fig = plot_equity_with_bh(run_id, strategy)

        prefix = {"c2c": "c2c", "o2c": "o2c", "bh": "bh"}.get(strategy, "c2c")

        def fmt(x, kind="float"):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "-"
            if kind == "pct":
                return f"{100*x:.2f}%"
            if kind == "int":
                return f"{int(x)}"
            return f"{x:.3f}"

        def get_metric(name):
            return row.get(f"{prefix}_{name}", None)

        sharpe   = get_metric("sharpe")
        cumret   = get_metric("cumulative_return")     # using your strategy-prefix metrics
        mdd      = get_metric("max_drawdown")
        turnover = get_metric("turnover")
        tau_star = row.get("tau_star", None)           # not prefix

        # stats table: keep it compact, but include the strategy-specific fields too
        stats_cols = [
            "intraday_freq", "cutoff", "selection_years", "grid_size", "rf", "transaction_cost",
            "tau_q_lo", "tau_q_hi", "tau_star",
            f"{prefix}_cumulative_return",
            f"{prefix}_sharpe",
            f"{prefix}_max_drawdown",
            f"{prefix}_turnover",
            f"{prefix}_num_buy_days",
            f"{prefix}_sharpe_buy_regime",
            f"{prefix}_sharpe_no_buy_regime",
        ]
        stats_cols = [c for c in stats_cols if c in rs.columns]
        stats = [{"metric": c, "value": str(row.get(c))} for c in stats_cols]

        # tau plot
        if TH_MAP is None:
            tau_fig = px.line(title="No thresholds table provided.")
            tau_help = "No thresholds loaded."
        else:
            th_run = TH_MAP.get(run_id)
            if th_run is None or th_run.empty:
                tau_fig = px.line(title="No τ diagnostics for this run.")
                tau_help = "No thresholds rows for this run."
            else:
                if ("tau" in th_run.columns) and ("score" in th_run.columns):
                    th_run = th_run.sort_values("tau")
                    tau_fig = px.line(th_run, x="tau", y="score")
                    if pd.notna(tau_star):
                        tau_fig.add_vline(x=float(tau_star), line_dash="dash")
                    tau_help = "Line = score over τ grid; dashed line = τ*."
                elif "tau" in th_run.columns:
                    tau_fig = px.histogram(th_run, x="tau", nbins=30)
                    if pd.notna(tau_star):
                        tau_fig.add_vline(x=float(tau_star), line_dash="dash")
                    tau_help = "Histogram = τ values; dashed line = τ*."
                else:
                    tau_fig = px.line(title="thresholds needs a 'tau' column.")
                    tau_help = "Add a 'tau' column (and optionally 'score')."
                tau_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="τ")

        # IMPORTANT: exactly 9 outputs returned
        return (
            eq_fig,
            fmt(sharpe),
            fmt(cumret, "pct"),
            fmt(mdd, "pct"),
            fmt(turnover),
            fmt(tau_star),
            stats,
            tau_fig,
            tau_help,
        )


    return app
