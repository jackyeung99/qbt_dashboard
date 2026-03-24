from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html
from common.plots import fmt


PAL = {
    "bh": "#0f172a",        # slate (buy&hold)
    "strategy": "#dc2626",  # near-black navy
    "pos": "#16a34a",       # green
    "neg": "#dc2626",       # red
    "weight": "#7c3aed",    # violet
    "state": "#f59e0b",     # amber
    "tau": "#06b6d4",       # cyan
    "grid": "rgba(15,23,42,0.08)",
    "regime_buy": "rgba(100,100,100,0.08)",   # soft green band
}



# ============================================================
# 2) Plot helpers
# ============================================================

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def _add_regime_shading(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    xcol: str = "date",
    signal_col: str = "signal",
) -> None:
    if xcol not in df.columns or signal_col not in df.columns or df.empty:
        return

    x = df[xcol]
    sig = pd.to_numeric(df[signal_col], errors="coerce").ffill()
    sig = sig.where(sig.isin([0, 1]))

    valid = sig.notna()
    if not valid.any():
        return

    x = x.loc[valid]
    sig = sig.loc[valid].astype(int)

    if len(x) == 0:
        return

    shapes = []
    start = x.iloc[0]
    prev = int(sig.iloc[0])

    for ts, s in zip(x.iloc[1:], sig.iloc[1:]):
        s = int(s)
        if s != prev:
            fill = "rgba(239,68,68,0.08)" if prev == 1 else "rgba(34,197,94,0.08)"
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start,
                    x1=ts,
                    y0=0,
                    y1=1,
                    fillcolor=fill,
                    line=dict(width=0),
                    layer="below",
                )
            )
            start = ts
            prev = s

    # final segment
    fill = "rgba(239,68,68,0.08)" if prev == 1 else "rgba(34,197,94,0.08)"
    shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=start,
            x1=x.iloc[-1],
            y0=0,
            y1=1,
            fillcolor=fill,
            line=dict(width=0),
            layer="below",
        )
    )

    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing_shapes + shapes)

def get_allocation_with_cash(df: pd.DataFrame) -> pd.Series:
    weight_cols = [c for c in df.columns if c.endswith("_weight")]
    if not weight_cols:
        return pd.Series(dtype=float)

    w = df[weight_cols].iloc[-1].fillna(0.0).astype(float)
    w.index = [c[:-len("_weight")] for c in weight_cols]

    invested = float(w.sum())
    cash = max(0.0, 1.0 - invested)

    if cash > 0:
        w.loc["Cash"] = cash

    return w


PLOT_THEME = {
    "bg": "white",
    "grid": "#E9EDF5",
    "axis": "#94A3B8",
    "text": "#0F172A",
    "muted": "#475569",
    "legend_bg": "rgba(255,255,255,0.85)",
    "shadow": "rgba(15,23,42,0.04)",
}


def _base_layout(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int = 420,
    margin: dict | None = None,
    show_legend: bool = True,
) -> go.Figure:
    if margin is None:
        margin = dict(l=24, r=24, t=64, b=24)

    fig.update_layout(
        template="simple_white",
        paper_bgcolor=PLOT_THEME["bg"],
        plot_bgcolor=PLOT_THEME["bg"],
        height=height,
        margin=margin,
        title=dict(
            text=title or "",
            x=0.01,
            xanchor="left",
            font=dict(size=20, color=PLOT_THEME["text"]),
        ),
        font=dict(
            family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
            size=12,
            color=PLOT_THEME["text"],
        ),
        legend=dict(
            orientation="h",
            y=1.10,
            yanchor="bottom",
            x=0.0,
            xanchor="left",
            bgcolor=PLOT_THEME["legend_bg"],
            borderwidth=0,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#CBD5E1",
            font=dict(color=PLOT_THEME["text"]),
        ),
        showlegend=show_legend,
    )

    # fig.update_xaxes(
    #     showgrid=True,
    #     gridcolor=PLOT_THEME["grid"],
    #     zeroline=False,
    #     linecolor=PLOT_THEME["axis"],
    #     tickfont=dict(color=PLOT_THEME["muted"]),
    #     title_font=dict(color=PLOT_THEME["muted"]),
    # )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOT_THEME["grid"],
        zeroline=False,
        linecolor=PLOT_THEME["axis"],
        tickfont=dict(color=PLOT_THEME["muted"]),
        title_font=dict(color=PLOT_THEME["muted"]),
    )
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color=PLOT_THEME["muted"]),
    )
    _base_layout(fig, height=320, show_legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def plot_allocation_pie(df: pd.DataFrame) -> go.Figure:
    alloc = get_allocation_with_cash(df)
    alloc = alloc[alloc > 0]

    if alloc.empty:
        return _empty_fig("No allocation data available.")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=alloc.index,
                values=alloc.values,
                hole=0.55,
                sort=False,
                direction="clockwise",
                textinfo="label+percent",
                textposition="outside",
                insidetextorientation="horizontal",
                marker=dict(line=dict(color="white", width=2)),
                hovertemplate="<b>%{label}</b><br>Allocation: %{percent}<extra></extra>",
            )
        ]
    )

    _base_layout(fig, title="Current Allocation", height=420, show_legend=False)
    fig.update_traces(
        pull=[0.02] * len(alloc),
    )
    fig.add_annotation(
        text="Allocation",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=PLOT_THEME["muted"]),
    )
    return fig


def plot_allocation_bar(df: pd.DataFrame) -> go.Figure:
    alloc = get_allocation_with_cash(df)
    alloc = alloc.sort_values(ascending=False)

    if alloc.empty:
        return _empty_fig("No allocation data available.")

    fig = go.Figure(
        go.Bar(
            x=alloc.index,
            y=alloc.values,
            text=[f"{v:.1%}" for v in alloc.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Allocation: %{y:.2%}<extra></extra>",
            marker=dict(
                line=dict(width=0),
            ),
        )
    )

    _base_layout(fig, title="Current Allocation", height=420, show_legend=False)
    fig.update_yaxes(title_text="Weight", tickformat=".0%")
    fig.update_xaxes(title_text="")
    return fig


def plot_rv_tau_weights_returns_equity(
    etf_df: pd.DataFrame,
    *,
    etf: str,
    date_col: str = "date",
    ret_col: str = "etf_ret",
    returns_as_bars: bool = True,
    lock_xticks: bool = True,
    pal: Dict[str, str] | None = None,
) -> go.Figure:
    palette = dict(PAL)
    if pal:
        palette.update(pal)

    required = [
        date_col,
        "weight",
        "_state_var",
        "_tau_star",
        ret_col,
        "etf_equity",
        "etf_bh_equity",
        "etf_bh_ret",
    ]
    missing = [c for c in required if c not in etf_df.columns]
    if missing:
        return _empty_fig(f"Missing required column(s): {', '.join(missing)}")

    df = etf_df.copy().sort_values(date_col)
    x = pd.to_datetime(df[date_col])
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    bh_ret = pd.to_numeric(df["etf_bh_ret"], errors="coerce")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.42, 0.18, 0.16, 0.24],
        subplot_titles=(
            f"{etf} Strategy vs Buy & Hold",
            f"{etf} Daily Returns",
            f"{etf} Target Weight",
            f"{etf} State Variable vs Threshold",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["etf_bh_equity"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=palette["bh"], width=2, dash="dash"),
            hovertemplate="<b>Buy & Hold</b><br>%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["etf_equity"],
            mode="lines",
            name="Strategy",
            line=dict(color=palette["strategy"], width=3),
            hovertemplate="<b>Strategy</b><br>%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if returns_as_bars:
        colors = np.where(bh_ret >= 0, palette["pos"], palette["neg"])
        fig.add_trace(
            go.Bar(
                x=x,
                y=bh_ret,
                name="Buy & Hold Return",
                marker_color=colors,
                opacity=0.9,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Return: %{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ret,
                mode="lines",
                name="ETF Return",
                line=dict(color=palette["strategy"], width=1.8),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Return: %{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["weight"],
            mode="lines",
            name="Weight",
            line=dict(color=palette["weight"], width=2.5),
            hovertemplate="<b>Weight</b><br>%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["_state_var"],
            mode="lines",
            name="State Variable",
            line=dict(color=palette["state"], width=2.5),
            hovertemplate="<b>State Variable</b><br>%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["_tau_star"],
            mode="lines",
            name="τ*",
            line=dict(color=palette["tau"], width=2, dash="dot"),
            hovertemplate="<b>τ*</b><br>%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    if "signal" in df.columns:
        _add_regime_shading(fig, df, xcol=date_col, signal_col="signal")

    _base_layout(
        fig,
        title=f"{etf} Regime Diagnostics",
        height=960,
        margin=dict(l=28, r=24, t=86, b=28),
        show_legend=True,
    )

    fig.update_annotations(font=dict(size=13, color=PLOT_THEME["text"]))

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1, tickformat=".1%")
    fig.update_yaxes(title_text="Weight", row=3, col=1, tickformat=".0%")
    fig.update_yaxes(title_text="State and τ*", row=4, col=1)

    # -------------------
    # Hide x-axis on top 3 rows
    # -------------------
    for r in [1, 2, 3]:
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    # -------------------
    # Bottom axis (weekly, clean)
    # -------------------
    fig.update_xaxes(
        title_text="Date",
        dtick=7 * 24 * 60 * 60 * 1000,   # weekly spacing
        tickformat="%b %d",              # e.g. Mar 18
        ticklabelmode="period",
        row=4,
        col=1,
    )

    if lock_xticks:
        fig.update_xaxes(nticks=8, row=4, col=1)

    return fig


def plot_portfolio(df: pd.DataFrame) -> go.Figure:
    required = ["session_date", "portfolio_value", "bh_equity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_fig(f"Missing required column(s): {', '.join(missing)}")

    dfx = df.copy().sort_values("session_date")
    dfx["session_date"] = pd.to_datetime(dfx["session_date"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dfx["session_date"],
            y=dfx["bh_equity"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=PAL["bh"], width=2, dash="dash"),
            hovertemplate="<b>Buy & Hold</b><br>%{x|%Y-%m-%d}<br>Equity: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dfx["session_date"],
            y=dfx["portfolio_value"],
            mode="lines",
            name="Portfolio",
            line=dict(color=PAL["strategy"], width=3),
            hovertemplate="<b>Portfolio</b><br>%{x|%Y-%m-%d}<br>Value: %{y:,.2f}<extra></extra>",
        )
    )

    _base_layout(
        fig,
        title="Portfolio vs SPY",
        height=420,
        margin=dict(l=24, r=24, t=72, b=24),
        show_legend=True,
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            x=1.0,
            xanchor="right",
            y=1.0,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
            borderwidth=0,
            font=dict(size=11),
        )
    )

    fig.update_yaxes(title_text="Equity")
    fig.update_xaxes(title_text="")
    return fig

def plot_avg_weights_from_metrics(metrics: dict) -> go.Figure:
    rows = []

    for k, v in metrics.items():
        if k.startswith("avg_") and k.endswith("_weight"):
            asset = k.replace("avg_", "").replace("_weight", "").upper()
            val = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if pd.notna(val):
                rows.append((asset, float(val)))

    if not rows:
        return _empty_fig("No average weight data available.")

    df_plot = pd.DataFrame(rows, columns=["asset", "avg_weight"])
    df_plot = df_plot[df_plot["avg_weight"] > 0]
    df_plot = df_plot.sort_values("avg_weight", ascending=True)

    if df_plot.empty:
        return _empty_fig("No positive average weights available.")

    fig = go.Figure(
        go.Bar(
            x=df_plot["avg_weight"],
            y=df_plot["asset"],
            orientation="h",
            text=[f"{v:.1%}" for v in df_plot["avg_weight"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Average Weight: %{x:.2%}<extra></extra>",
            marker=dict(
                line=dict(width=0),
            ),
        )
    )

    _base_layout(
        fig,
        title="Average Allocation by Asset",
        height=max(320, 40 * len(df_plot) + 140),
        margin=dict(l=32, r=32, t=64, b=32),
        show_legend=False,
    )

    fig.update_xaxes(
        title_text="Average Weight",
        tickformat=".0%",
        range=[0, max(df_plot["avg_weight"].max() * 1.15, 0.05)],
    )
    fig.update_yaxes(
        title_text="",
        categoryorder="array",
        categoryarray=df_plot["asset"],
    )

    return fig






def metric_row(label: str, value: str) -> html.Div:
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr auto",
            "columnGap": "12px",
            "padding": "8px 0",
            "borderBottom": "1px solid #f0f0f0",
        },
        children=[
            html.Div(label, style={"color": "#666"}),
            html.Div(value, style={"fontWeight": 700}),
        ],
    )

def build_signal_table_card(etf_df: pd.DataFrame, sig_metrics: dict) -> html.Div:
    current_state = etf_df["_state_var"].iloc[-1] if "_state_var" in etf_df.columns and not etf_df.empty else np.nan
    current_tau = etf_df["_tau_star"].iloc[-1] if "_tau_star" in etf_df.columns and not etf_df.empty else np.nan
    current_signal = etf_df["signal"].iloc[-1] if "signal" in etf_df.columns and not etf_df.empty else np.nan

    # signal_strength = (
    #     current_state - current_tau
    #     if pd.notna(current_state) and pd.notna(current_tau)
    #     else np.nan
    # )
    low_vol_target_weight = etf_df.iloc[-1]['_w_low']
    high_vol_target_weight = etf_df.iloc[-1]['_w_high']

   
    current_regime_label = "-"
    if pd.notna(current_signal):
        current_regime_label = "High-Volatility Regime" if int(current_signal) == 1 else "Low-Volatility Regime"

    previous_regime = sig_metrics.get("signal_previous_regime")
    previous_regime_label = "-"
    if previous_regime is not None:
        previous_regime_label = "High-Volatility Regime" if int(previous_regime) == 1 else "Low-Volatility Regime"

    rows = [
        ("Current Regime", current_regime_label),
        ("Previous Regime", previous_regime_label),
        ("Current State Var", fmt(current_state, decimals=4)),
        ("Current Tau*", fmt(current_tau, decimals=4)),
        ("Low-Regime Target Weight", fmt(low_vol_target_weight, style="pct", decimals=1)),
        ("High-Regime Target Weight", fmt(high_vol_target_weight, style="pct", decimals=1)),
        # ("Signal Strength", fmt(signal_strength, decimals=4)),
        ("Days in Current Regime", fmt(sig_metrics.get("signal_days_in_current_regime"), decimals=0)),
        ("% Time High Regime", fmt(sig_metrics.get("signal_pct_state_1"), style="pct", decimals=1)),
        ("% Time Low Regime", fmt(sig_metrics.get("signal_pct_state_0"), style="pct", decimals=1)),
        ("Signal Flips", fmt(sig_metrics.get("signal_n_flips"), decimals=0)),
        ("Flips / Year", fmt(sig_metrics.get("signal_flips_per_year"), decimals=2)),
        ("Avg Regime Duration", fmt(sig_metrics.get("signal_avg_regime_duration"), decimals=1)),
        ("Avg High-Regime Duration", fmt(sig_metrics.get("signal_avg_hold_state_1"), decimals=1)),
        ("Avg Low-Regime Duration", fmt(sig_metrics.get("signal_avg_hold_state_0"), decimals=1)),
        ("Longest High-Regime Streak", fmt(sig_metrics.get("signal_max_hold_state_1"), decimals=0)),
        ("Longest Low-Regime Streak", fmt(sig_metrics.get("signal_max_hold_state_0"), decimals=0)),
        ("0 → 1 Transitions", fmt(sig_metrics.get("signal_n_0_to_1"), decimals=0)),
        ("1 → 0 Transitions", fmt(sig_metrics.get("signal_n_1_to_0"), decimals=0)),
    ]


    return html.Div(
        children=[
            html.Div(
                "Regime Diagnostics",
                style={"fontSize": "16px", "fontWeight": 700, "marginBottom": "10px"},
            ),
            *[metric_row(label, value) for label, value in rows],
        ]
    )

def build_portfolio_table_card(metrics: dict) -> html.Div:
    rows = [
        ("Total Return", fmt(metrics.get("total_return"), style='pct', decimals=2)),
        ("Strategy Sharpe", fmt(metrics.get("sharpe"), decimals=2)),
        ("Benchmark Sharpe (S&P 500)", fmt(metrics.get("bh_sharpe"), decimals=2)),
        ("Sharpe Alpha vs Benchmark", fmt(metrics.get("sharpe_minus_bh"), decimals=2)),
        ("CAGR", fmt(metrics.get("cagr"), style="pct", decimals=2)),
        ("Max Drawdown", fmt(metrics.get("max_dd"), style="pct", decimals=2)),
        ("Avg Daily Volatility", fmt(metrics.get("vol_ann"), style="pct", decimals=2)),
        ("Avg Daily Return", fmt(metrics.get("mean"), style="pct", decimals=3)),
        ("Avg Daily Gain", fmt(metrics.get("avg_gain"), style="pct", decimals=3)),
        ("Avg Daily Loss", fmt(metrics.get("avg_loss"), style="pct", decimals=3))
        
    ]


    return html.Div(
        children=[
            html.Div(
                "Portfolio Statistics",
                style={"fontSize": "16px", "fontWeight": 700, "marginBottom": "10px"},
            ),
            *[metric_row(label, value) for label, value in rows],
        ]
    )

