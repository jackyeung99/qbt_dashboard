from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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
# 1) Normalization / prep helpers (no classes)
# ============================================================

def _ensure_session_date(df: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    x = df.copy()
    if date_col in x.columns:
        x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
        x = x.dropna(subset=[date_col]).sort_values(date_col)
    else:
        x.index = pd.to_datetime(x.index, errors="coerce")
        x = x.rename_axis(date_col).reset_index()
        x = x.dropna(subset=[date_col]).sort_values(date_col)

    if not x.empty:
        x = x.drop_duplicates(date_col, keep="last").sort_values(date_col)

    return x


def _coerce_numeric(df: pd.DataFrame, cols: List[str], *, ffill: bool = False) -> pd.DataFrame:
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
            if ffill:
                x[c] = x[c].ffill()
    return x


def _equity_from_returns(r: pd.Series, *, returns_are_log: bool) -> pd.Series:
    r = pd.to_numeric(r, errors="coerce").fillna(0.0).astype(float)
    if returns_are_log:
        return np.exp(r.cumsum())
    return (1.0 + r).cumprod()


def _normalize_to_one(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.empty:
        return s
    s0 = float(s.iloc[0])
    if not np.isfinite(s0) or s0 == 0:
        return pd.Series(np.nan, index=s.index)
    return s / s0


def normalize_portfolio(
    equity: pd.DataFrame | pd.Series,
    *,
    date_col: str = "session_date",
    portfolio_value_col: str = "portfolio_value",
    returns_are_log: bool = True,
) -> pd.DataFrame:
    """
    Outputs a DataFrame with:
      - session_date
      - portfolio_value
      - strategy_equity_norm   (from portfolio_value)
      - bh_equity, bh_equity_norm (from bh_ret_col)
      - weight, turnover
      - state_var
      - strategy_growth        (from ret_col if present)
    """

    if equity is None:
        return pd.DataFrame()

    if isinstance(equity, pd.Series):
        equity = equity.to_frame(name=portfolio_value_col)

    x = equity.copy()
    x = _ensure_session_date(x, date_col=date_col)

    # portfolio value (fallback)
    if portfolio_value_col not in x.columns:
        if "equity" in x.columns:
            x = x.rename(columns={"equity": portfolio_value_col})
        else:
            raise ValueError(f"Missing {portfolio_value_col!r} (and no 'equity' fallback).")



    initial_value = float(x[portfolio_value_col].iloc[0])

    r = x["SPY_ret_cc"].fillna(0.0).astype(float)

    x["bh_equity"] = initial_value * np.exp(r.cumsum().shift(fill_value=0))
    
    return x 
    # return x.dropna(subset=['trained_at_utc'])


# ============================================================
# 2) Plot helpers
# ============================================================

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def _add_regime_shading(fig: go.Figure, df: pd.DataFrame, *, xcol: str = "session_date", signal_col: str = "XLE_weight") -> None:
    if signal_col not in df.columns:
        return

    x = df[xcol]
    sig = pd.to_numeric(df[signal_col], errors="coerce").ffill()
    sig = sig.where(sig.isin([0, 1]))
    if not sig.notna().any():
        return

    shapes = []
    start = x.iloc[0]
    prev = int(sig.iloc[0]) if pd.notna(sig.iloc[0]) else None

    for ts, s in zip(x.iloc[1:], sig.iloc[1:]):
        if pd.isna(s):
            continue
        s = int(s)
        if prev is not None and s != prev:
            fill = "rgba(34,197,94,0.08)" if prev == 1 else "rgba(239,68,68,0.08)"
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
        elif prev is None:
            start = ts
            prev = s

    if prev is not None:
        fill = "rgba(34,197,94,0.08)" if prev == 1 else "rgba(239,68,68,0.08)"
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

    fig.update_layout(shapes=shapes)


def _build_frames(
    *,
    df: pd.DataFrame,
    xcol: str,
    series: List[Tuple[str, str, str]],  # (trace_type, ycol, extra) where trace_type in {"line","bar"}
    every: int,
    frame_ms: int,
    pal: Dict[str, str],
    returns_as_bars: bool,
    use_webgl: bool,
) -> Tuple[List[go.Frame], List[pd.Timestamp]]:
    ScatterLine = go.Scattergl if use_webgl else go.Scatter
    x = df[xcol]
    timeline = x.iloc[::every].to_list()
    frames: List[go.Frame] = []

    for i, t in enumerate(timeline):
        mask = x <= t
        data_updates = []

        for trace_type, ycol, role in series:
            if ycol not in df.columns:
                continue

            y = df[ycol]

            if trace_type == "bar":
                y_masked = y[mask]
                # only used for returns panel
                colors = np.where(y_masked >= 0, pal["pos"], pal["neg"]).tolist()
                data_updates.append(
                    go.Bar(x=x[mask], y=y_masked, marker_color=colors, name=role)
                )
            else:
                data_updates.append(dict(x=x[mask], y=y[mask]))

        frames.append(go.Frame(name=str(i), data=data_updates))

    return frames, timeline


# ============================================================
# 3) Main plot function
# ============================================================



def get_etf_view(
    df: pd.DataFrame,
    etf: str,
    *,
    date_col: str = "session_date",
) -> pd.DataFrame:
    prefix = f"{etf}_"

    etf_cols = [c for c in df.columns if c.startswith(prefix)]
    if not etf_cols:
        raise ValueError(f"No columns found for ETF {etf!r}")

    keep = [date_col] + etf_cols if date_col in df.columns else etf_cols

    out = df[keep].copy()

    renamed = {
        c: c[len(prefix):]
        for c in etf_cols
    }
    out = out.rename(columns=renamed)

    if date_col in out.columns and date_col != "date":
        out = out.rename(columns={date_col: "date"})

    return out

def plot_rv_tau_weights_returns_equity(
    base_df: pd.DataFrame,
    *,
    etf: str,
    run_id: str | None = None,
    returns_as_bars: bool = True,
    lock_xticks: bool = True,
    pal: Dict[str, str] | None = None,
    debug: bool = False,
) -> go.Figure:
    if pal:
        PAL.update(pal)

    if base_df is None or base_df.empty:
        return _empty_fig("No data.")

    try:
        df = get_etf_view(base_df, etf)
    except ValueError as e:
        return _empty_fig(str(e))

    if "date" not in df.columns:
        return _empty_fig("Missing required column: date")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if "ret_cc" in df.columns:
        ret_col = "ret_cc"
    elif "ret" in df.columns:
        ret_col = "ret"
    else:
        ret_col = None

    required = ["weight", "_state_var", "_tau_star", ret_col]
    required = [c for c in required if c is not None]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_fig(f"Missing required column(s): {', '.join(missing)}")

    numeric_cols = ["weight", "_state_var", "_tau_star", ret_col, "signal", "_w_high", "_w_low"]
    for c in numeric_cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["weight"] = df["weight"].ffill().fillna(0.0)
    df["_state_var"] = pd.to_numeric(df["_state_var"]).ffill()
    df["_tau_star"] = pd.to_numeric(df["_tau_star"]).ffill()

    mask = df["_state_var"].notna() & df["_tau_star"].notna()

    df["signal"] = 0
    df.loc[mask, "signal"] = (df.loc[mask, "_state_var"] > df.loc[mask, "_tau_star"]).astype(int)
        
    ret = df[ret_col].fillna(0.0)

    # strategy sleeve return = dynamic weight * ETF return
    df["strategy_ret"] = df["weight"] * ret

    # buy-and-hold sleeve = invest w_max the whole time
    if "_w_high" in df.columns:
        w_max = float(pd.to_numeric(df["_w_high"], errors="coerce").dropna().max())
    else:
        w_max = float(pd.to_numeric(df["weight"], errors="coerce").dropna().max())

    if not np.isfinite(w_max):
        w_max = 0.0

    df["bh_ret"] = w_max * ret

    # equity curves from CC returns
    df["strategy_equity"] = np.exp(df["strategy_ret"].cumsum().shift(fill_value=0.0))
    df["bh_equity"] = np.exp(df["bh_ret"].cumsum().shift(fill_value=0.0))

    x = df["date"]
    xmin, xmax = x.min(), x.max()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.20, 0.18, 0.30],
        subplot_titles=(
            f"{etf} Equity",
            f"{etf} Returns",
            f"Weight ({etf})",
            f"{etf} State Variable and τ*",
        ),
    )

    # Row 1: derived equities
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["bh_equity"],
            mode="lines",
            name=f"Buy & Hold at w_max={w_max:.2f}",
            line=dict(dash="dash", color=PAL["bh"], width=2),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["strategy_equity"],
            mode="lines",
            name="Strategy sleeve equity",
            line=dict(color=PAL["strategy"], width=3),
        ),
        row=1, col=1,
    )

    # Row 2: raw ETF returns
    if returns_as_bars:
        colors = np.where(ret >= 0, PAL["pos"], PAL["neg"]).tolist()
        fig.add_trace(
            go.Bar(
                x=x,
                y=ret,
                name="ETF return",
                marker_color=colors,
                opacity=0.85,
            ),
            row=2, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ret,
                mode="lines",
                name="ETF return",
                line=dict(width=2),
            ),
            row=2, col=1,
        )

    # Row 3: weight
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["weight"],
            mode="lines",
            name="Weight",
            line=dict(color=PAL["weight"], width=2),
        ),
        row=3, col=1,
    )

    if "_w_high" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["_w_high"],
                mode="lines",
                name="w_high",
                line=dict(color=PAL["tau"], width=1, dash="dot"),
                opacity=0.7,
            ),
            row=3, col=1,
        )

    if "_w_low" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["_w_low"],
                mode="lines",
                name="w_low",
                line=dict(color=PAL["tau"], width=1, dash="dot"),
                opacity=0.7,
            ),
            row=3, col=1,
        )

    # Row 4: state + tau
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["_state_var"],
            mode="lines",
            name="State variable",
            line=dict(color=PAL["state"], width=2),
        ),
        row=4, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["_tau_star"],
            mode="lines",
            name="τ*",
            line=dict(color=PAL["tau"], width=2, dash="dot"),
        ),
        row=4, col=1,
    )

    if "signal" in df.columns:
        _add_regime_shading(fig, df, xcol="date", signal_col="signal")

    fig.update_xaxes(range=[xmin, xmax], autorange=False, showgrid=True, zeroline=False)
    fig.update_xaxes(
        tickmode="linear",
        dtick="M3",
        tickformat="%Y-%m",
        showticklabels=True,
        automargin=True,
        row=4,
        col=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        height=920,
        margin=dict(l=20, r=20, t=90, b=40),
        legend=dict(
            orientation="h",
            y=1.02,
            yanchor="bottom",
            x=0.0,
            xanchor="left",
            font=dict(size=12),
        ),
        uirevision=f"run:{run_id}" if run_id else "lock",
        barmode="relative",
    )

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="ETF return", row=2, col=1)
    fig.update_yaxes(title_text="Weight", row=3, col=1)
    fig.update_yaxes(title_text="State / τ*", row=4, col=1)
    fig.update_xaxes(title_text="Time", title_standoff=18, row=4, col=1)

    if lock_xticks:
        fig.update_xaxes(nticks=6)

    return fig


def plot_portfolio(dfr: pd.DataFrame):

    fig = px.line(
        dfr,
        x="session_date",
        y=["portfolio_value", "bh_equity"],  # <- multiple series
        title="Portfolio vs Buy & Hold"
    )

    return fig



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

def plot_allocation_pie(df: pd.DataFrame):
    alloc = get_allocation_with_cash(df)
    alloc = alloc[alloc > 0]

    fig = px.pie(
        values=alloc.values,
        names=alloc.index,
        title="Current Allocation",
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.2%}<extra></extra>",
    )

    return fig


def plot_allocation_bar(df: pd.DataFrame):
    alloc = get_allocation_with_cash(df)

    fig = px.bar(
        x=alloc.index,
        y=alloc.values,
        title="Current Allocation",
    )

    fig.update_traces(
        hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"
    )

    fig.update_layout(
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig