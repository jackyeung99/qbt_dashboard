from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html


# =============================================================================
# Paths + Cached single-file loaders (Cloud-friendly)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[0]
SWEEP_DIR = BASE_DIR / "results" / "xle_rv_sweep"


@lru_cache(maxsize=1)
def load_equity_all() -> pd.DataFrame:
    """Load ALL equity curves once per worker (fast sequential parquet read)."""
    df = pd.read_parquet(SWEEP_DIR / "equity_curves.parquet", columns=["run_id", "strategy", "timestamp", "equity"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # categories shrink memory + speed filters
    df["run_id"] = df["run_id"].astype("category")
    df["strategy"] = df["strategy"].astype("category")
    # sorted improves slicing locality
    return df.sort_values(["run_id", "strategy", "timestamp"])


@lru_cache(maxsize=1)
def load_returns_all() -> pd.DataFrame:
    """Load ALL returns once per worker (then slice by run_id in memory)."""
    cols = ["run_id", "timestamp", "split", "rvol_o2c", "ret_cc", "ret_oc"]
    df = pd.read_parquet(SWEEP_DIR / "returns.parquet", columns=cols)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["run_id"] = df["run_id"].astype("category")
    if "split" in df.columns:
        df["split"] = df["split"].astype("category")
    return df.sort_values(["run_id", "timestamp"])


# =============================================================================
# UI helpers
# =============================================================================

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


def options_from_unique(s: pd.Series) -> List[dict]:
    vals = sorted([v for v in s.dropna().unique().tolist()])
    return [{"label": str(v), "value": v} for v in vals]


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


# =============================================================================
# Plotting
# =============================================================================

def plot_tau_diagnostics(th_run: pd.DataFrame, tau_star: float) -> go.Figure:
    th_run = th_run.sort_values("tau").copy()
    tau_star = float(tau_star)

    fig = px.line(
        th_run,
        x="tau",
        y="sharpe_diff",
        labels={"tau": r"Realized Volatility Threshold $\tau$", "sharpe_diff": r"$\Delta$ Sharpe"},
        template="simple_white",
    )
    fig.update_traces(line=dict(color="black", width=3), name="Δ Sharpe")

    if "sharpe_low_state" in th_run.columns:
        fig.add_trace(
            go.Scatter(
                x=th_run["tau"],
                y=th_run["sharpe_low_state"],
                mode="lines",
                name="Sharpe (Low Vol Regime)",
                line=dict(width=1, dash="dot"),
                opacity=0.6,
            )
        )
    if "sharpe_high_state" in th_run.columns:
        fig.add_trace(
            go.Scatter(
                x=th_run["tau"],
                y=th_run["sharpe_high_state"],
                mode="lines",
                name="Sharpe (High Vol Regime)",
                line=dict(width=1, dash="dot"),
                opacity=0.6,
            )
        )

    fig.add_vline(
        x=tau_star,
        line_width=3,
        line_dash="dash",
        line_color="black",
        annotation_text="τ*",
        annotation_position="top",
        annotation=dict(
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


def plot_rv_vs_returns(ret_run: pd.DataFrame, *, rv_col: str, ret_col: str, tau_star: float) -> go.Figure:
    x = ret_run.copy()

    if {"series", "value"}.issubset(x.columns):
        x = (
            x.pivot_table(index=["timestamp", "split"], columns="series", values="value", aggfunc="first")
            .reset_index()
        )

    need = {"split", rv_col, ret_col}
    missing = [c for c in need if c not in x.columns]
    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing columns for RV vs Returns plot: {missing}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(template="simple_white")
        return fig

    x["split"] = x["split"].astype(str)

    fig = px.scatter(
        x,
        x=rv_col,
        y=ret_col,
        color="split",
        opacity=0.7,
        template="simple_white",
        labels={rv_col: "Realized Volatility", ret_col: "Daily Returns XLE"},
    )

    fig.add_vline(
        x=float(tau_star),
        line_width=3,
        line_dash="dash",
        line_color="black",
        annotation_text="τ*",
        annotation_position="top",
    )

    means = x.groupby("split")[ret_col].mean().to_dict()

    def _rename(tr):
        if tr.name in means:
            m = means[tr.name]
            new = f"{tr.name} (mean={m:.3g})"
            tr.update(name=new, legendgroup=new)

    fig.for_each_trace(_rename)

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=60, r=20, t=40, b=50))
    return fig


# =============================================================================
# App factory
# =============================================================================

def make_app(
    runs_summary: pd.DataFrame,
    thresholds: Optional[pd.DataFrame] = None,
) -> Dash:
    rs = runs_summary.copy()
    if "run_id" not in rs.columns:
        raise ValueError("runs_summary must contain 'run_id'")

    if "run_label" not in rs.columns:
        rs["run_label"] = rs.apply(build_pretty_label, axis=1)

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

    TH_MAP: Optional[Dict[str, pd.DataFrame]] = None
    if thresholds is not None:
        th = thresholds.copy()
        if "run_id" not in th.columns:
            raise ValueError("thresholds must contain 'run_id'")
        TH_MAP = {rid: g for rid, g in th.groupby("run_id", sort=False)}

    rs_sorted = rs.sort_values("run_label")
    rs_default = rs_sorted.head(200)
    default_opts = [{"label": r["run_label"], "value": r["run_id"]} for r in rs_default.to_dict("records")]
    default_value = default_opts[0]["value"] if default_opts else None

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "system-ui", "padding": "16px", "background": "#f6f7fb"},
        children=[
            html.H2("Tau Sensitivity Dashboard", style={"margin": "0 0 10px 0"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "12px", "minWidth": 0},
                children=[
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
                            html.Div(id="filtered-count", style={"marginTop": "10px", "fontSize": "12px", "color": "#666"}),
                        ],
                    ),
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
                                        style={"background": "white", "borderRadius": "12px", "padding": "12px", "border": "1px solid #e6e6e6", "minWidth": 0},
                                        children=[
                                            html.Div("Equity Curve", style={"fontWeight": 700, "marginBottom": "6px"}),
                                            dcc.Graph(id="equity-fig", config={"displayModeBar": False, "responsive": True}, style={"height": "420px"}),
                                        ],
                                    ),
                                    html.Div(
                                        style={"background": "white", "borderRadius": "12px", "padding": "12px", "border": "1px solid #e6e6e6", "minWidth": 0},
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
                                style={"marginTop": "12px", "background": "white", "borderRadius": "12px", "padding": "12px", "border": "1px solid #e6e6e6", "minWidth": 0},
                                children=[
                                    html.Div("τ Diagnostics", style={"fontWeight": 700, "marginBottom": "6px"}),
                                    html.Div(
                                        style={"display": "flex", "gap": "12px"},
                                        children=[
                                            html.Div(style={"flex": 1, "minWidth": 0}, children=[dcc.Graph(id="tau-fig", config={"displayModeBar": False}, style={"height": "320px"}), html.Div(id="tau-help", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"})]),
                                            html.Div(style={"flex": 1, "minWidth": 0}, children=[dcc.Graph(id="eval-test", config={"displayModeBar": False}, style={"height": "320px"}), html.Div(id="eval-help", style={"fontSize": "12px", "color": "#666", "marginTop": "6px"})]),
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

    def _empty_fig():
        return px.line(pd.DataFrame({"x": [], "y": []}), x="x", y="y")

    def _fmt(x, kind="float"):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        if kind == "pct":
            return f"{100 * x:.2f}%"
        return f"{x:.3f}"

    def filter_runs(filters: dict) -> pd.DataFrame:
        out = rs
        for k, v in filters.items():
            if v is None:
                continue
            out = out[out[k] == v]
        return out

    dropdown_inputs = [Input({"type": "param-dd", "name": c}, "value") for c in param_cols]

    @callback(
        Output("run-dd", "options"),
        Output("run-dd", "value"),
        Output("filtered-count", "children"),
        dropdown_inputs,
        prevent_initial_call=True,
    )
    def update_run_dropdown(*vals):
        filters = {c: v for c, v in zip(param_cols, vals)}
        dff = filter_runs(filters).sort_values("run_label").head(500)
        opts = [{"label": r["run_label"], "value": r["run_id"]} for r in dff.to_dict("records")]
        default_val = opts[0]["value"] if opts else None
        return opts, default_val, f"{len(dff)} run(s) in list (capped to 500)."

    def plot_equity(run_id: str, strategy: str) -> go.Figure:
        df = load_equity_all()
        g = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == "bh")]
        fig = go.Figure()
        if not g.empty:
            fig.add_trace(go.Scatter(x=g["timestamp"], y=g["equity"], mode="lines", name="Buy & Hold", line=dict(color="gray", dash="dash")))

        s = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == strategy)]
        if not s.empty and strategy != "bh":
            fig.add_trace(go.Scatter(x=s["timestamp"], y=s["equity"], mode="lines", name=strategy.upper(), line=dict(width=3)))

        if len(fig.data) == 0:
            fig.add_annotation(text="No equity data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Equity", legend=dict(orientation="h", y=1.02, yanchor="bottom"))
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
        Output("eval-test", "figure"),
        Output("eval-help", "children"),
        Input("run-dd", "value"),
        Input("strategy-radio", "value"),
        prevent_initial_call=False,
    )
    def render_run(run_id, strategy):
        empty_fig = _empty_fig()

        if not run_id:
            return empty_fig, "-", "-", "-", "-", "-", [], empty_fig, "Select a run.", empty_fig, "Select a run."

        row_df = rs.loc[rs["run_id"] == run_id]
        if row_df.empty:
            return empty_fig, "-", "-", "-", "-", "-", [], empty_fig, "Run not found.", empty_fig, "Run not found."
        row = row_df.iloc[0]

        eq_fig = plot_equity(run_id, strategy)

        prefix = {"c2c": "c2c", "o2c": "o2c", "bh": "bh"}.get(strategy, "c2c")
        sharpe = row.get(f"{prefix}_sharpe", None)
        cumret = row.get(f"{prefix}_cumulative_return", None)
        mdd = row.get(f"{prefix}_max_drawdown", None)
        turnover = row.get(f"{prefix}_turnover", None)
        tau_star = row.get("tau_star", None)

        stats_cols = [
            "intraday_freq", "cutoff", "selection_years", "grid_size", "rf", "transaction_cost",
            "tau_q_lo", "tau_q_hi", "tau_star",
            f"{prefix}_cumulative_return", f"{prefix}_sharpe", f"{prefix}_max_drawdown", f"{prefix}_turnover",
        ]
        stats_cols = [c for c in stats_cols if c in rs.columns]
        stats = [{"metric": c, "value": str(row.get(c))} for c in stats_cols]

        tau_fig = empty_fig
        tau_help = "No τ diagnostics."
        eval_fig = empty_fig
        eval_help = "No RV vs returns data."

        if TH_MAP is not None:
            th_run = TH_MAP.get(run_id)
            if th_run is not None and not th_run.empty and "tau" in th_run.columns:
                tau_fig = plot_tau_diagnostics(th_run, tau_star)
                tau_help = "ΔSharpe over τ grid; dashed line is τ*."

        # Returns slice (in-memory)
        ret = load_returns_all()
        ret_run = ret[ret["run_id"] == run_id]
        if not ret_run.empty:
            rv_col = "rvol_o2c"
            ret_col = "ret_cc" if prefix == "c2c" else "ret_oc"
            eval_fig = plot_rv_vs_returns(ret_run, rv_col=rv_col, ret_col=ret_col, tau_star=tau_star)
            eval_help = f"Scatter of {ret_col} vs {rv_col}. Color = split."

        return (
            eq_fig,
            _fmt(sharpe),
            _fmt(cumret, "pct"),
            _fmt(mdd, "pct"),
            _fmt(turnover),
            _fmt(tau_star),
            stats,
            tau_fig,
            tau_help,
            eval_fig,
            eval_help,
        )

    return app
