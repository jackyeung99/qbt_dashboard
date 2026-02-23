from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import plotly.graph_objects as go


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


def normalize_equity_df(
    equity: pd.DataFrame | pd.Series,
    *,
    date_col: str = "session_date",
    portfolio_value_col: str = "portfolio_value",
    state_var_col: str = "XLE_rvol",
    ret_col: str = "ret",
    bh_ret_col: str = "XLE_ret_cc",
    weight_col: str = "XLE_weight",
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

    # coerce core numeric
    x = _coerce_numeric(
        x,
        cols=[portfolio_value_col, weight_col, state_var_col, bh_ret_col, ret_col],
        ffill=False,
    )

    x = x.dropna(subset=[date_col, portfolio_value_col]).sort_values(date_col)
    if x.empty:
        return x

    # strategy growth from ret_col (optional)
    if ret_col in x.columns:
        r = pd.to_numeric(x[ret_col], errors="coerce").fillna(0.0).astype(float)
        x["strategy_growth"] = np.exp(r) if returns_are_log else (1.0 + r)
    else:
        x["strategy_growth"] = np.nan

    # weight + turnover
    if weight_col in x.columns:
        x["weight"] = pd.to_numeric(x[weight_col], errors="coerce").ffill().fillna(0.0).astype(float)
        x["turnover"] = float(x["weight"].diff().abs().sum())
    else:
        x["weight"] = np.nan
        x["turnover"] = 0.0

    # state variable
    x["state_var"] = pd.to_numeric(x.get(state_var_col, np.nan), errors="coerce").ffill()

    # buy & hold equity
    if bh_ret_col not in x.columns:
        raise ValueError(f"Buy&Hold return column not found: {bh_ret_col!r}")
    x["bh_equity"] = _equity_from_returns(x[bh_ret_col], returns_are_log=returns_are_log)

    # normalized curves
    x["strategy_equity_norm"] = _normalize_to_one(x[portfolio_value_col].astype(float))
    x["bh_equity_norm"] = _normalize_to_one(x["bh_equity"].astype(float))

    return x

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

def plot_rv_tau_weights_returns_equity_animated(
    dfr: pd.DataFrame,
    *,
    run_id: str | None = None,
    frame_ms: int = 200,
    every: int = 1,
    # column selection (matches your schema)
    state_var_col: str = "XLE_rvol",
    ret_col: str = "XLE_ret_cc",
    bh_ret_col: str = "XLE_ret_cc",
    weight_col: str = "XLE_weight",
    returns_are_log: bool = True,
    # rendering
    returns_as_bars: bool = True,
    use_webgl: bool = False,
    lock_xticks: bool = True,
    debug: bool = False,
    pal: Dict[str, str] | None = None,
) -> go.Figure:
    
    if pal:
        PAL.update(pal)

    if dfr is None or dfr.empty:
        return _empty_fig("No data.")

    # normalize/compute required plot columns from your raw schema
    df = normalize_equity_df(
        dfr,
        state_var_col=state_var_col,
        ret_col=ret_col,
        bh_ret_col=bh_ret_col,
        weight_col=weight_col,
        returns_are_log=returns_are_log,
    )

    if df.empty:
        return _empty_fig("No rows after cleaning.")

    # ensure plot-needed columns exist
    need = {"session_date", "strategy_equity_norm", "bh_equity_norm", "weight", "state_var"}
    missing = need - set(df.columns)
    if missing:
        return _empty_fig(f"Missing required columns after normalize: {', '.join(sorted(missing))}")

    # (Optional) coerce + ffill continuity for plotting (not returns)
    df = _coerce_numeric(df, ["strategy_equity_norm", "bh_equity_norm", "weight", "state_var"], ffill=True)
    if ret_col in df.columns:
        df = _coerce_numeric(df, [ret_col], ffill=False)

    x = df["session_date"]
    xmin, xmax = x.min(), x.max()

    if debug:
        print("\n[plot_debug] rows:", len(df), "xmin:", xmin, "xmax:", xmax)
        print("[plot_debug] cols:", list(df.columns))

    # figure scaffold
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.2, 0.18, 0.3],
        subplot_titles=(
            f"Normalized Equity (Strategy vs B&H: {bh_ret_col})",
            f"XLE Returns ({ret_col})" if ret_col in df.columns else "Strategy Returns (missing)",
            f"Weights / Exposure ({weight_col})",
            f"State Variable ({state_var_col})",
        ),
    )





    StaticScatter = go.Scattergl if use_webgl else go.Scatter
    AnimScatter = go.Scatter  # force

    trace_idx = {}
    sl0 = slice(0, 1)
    # --- BH (animated) ---
    fig.add_trace(
        AnimScatter(
            x=x.iloc[sl0], y=df["bh_equity_norm"].iloc[sl0],
            mode="lines",
            name=f"Buy & Hold ({bh_ret_col})",
            line=dict(dash="dash", color=PAL["bh"]),
        ),
        row=1, col=1,
    )
    trace_idx["bh"] = len(fig.data) - 1

    # --- Strategy equity (STATIC / full) ---
    fig.add_trace(
        StaticScatter(
            x=x, y=df["strategy_equity_norm"],
            mode="lines",
            name="XLE Volatility Strategy (normalized)",
            line=dict(width=3, color=PAL["strategy"]),
        ),
        row=1, col=1,
    )
    trace_idx["strategy"] = len(fig.data) - 1

    # --- Returns (animated) ---
    has_ret = ret_col in df.columns
    if has_ret:
        r0 = pd.to_numeric(df[ret_col], errors="coerce").iloc[sl0]
        if returns_as_bars:
            r0 = pd.to_numeric(df[ret_col], errors="coerce").iloc[sl0]
            colors0 = np.where(r0.fillna(0.0) >= 0, PAL["pos"], PAL["neg"]).tolist()

            fig.add_trace(
                go.Bar(
                    x=x.iloc[sl0],
                    y=r0,
                    name="Return",
                    marker_color=colors0,   # <<< consistent
                ),
                row=2, col=1,
            )
        else:
            fig.add_trace(AnimScatter(x=x.iloc[sl0], y=r0, mode="lines", name="Return"), row=2, col=1)
        trace_idx["ret"] = len(fig.data) - 1

    # --- Weight (animated) ---
    fig.add_trace(
        AnimScatter(
            x=x.iloc[sl0], y=df["weight"].iloc[sl0],
            mode="lines",
            name="Weight",
            line=dict(color=PAL["weight"], width=2),
        ),
        row=3, col=1,
    )
    trace_idx["weight"] = len(fig.data) - 1

    # --- State var (animated) ---
    fig.add_trace(
        AnimScatter(
            x=x.iloc[sl0], y=df["state_var"].iloc[sl0],
            mode="lines",
            name="State Var",
            line=dict(color=PAL["state"], width=2),
        ),
        row=4, col=1,
    )
    trace_idx["state"] = len(fig.data) - 1

    # --- Tau* (animated) ---
    fig.add_trace(
        AnimScatter(
            x=x.iloc[sl0], y=df["tau_star"].fillna(0).iloc[sl0],
            mode="lines",
            name="Tau*",
            line=dict(color=PAL["tau"], width=2),
        ),
        row=4, col=1,
    )
    trace_idx["tau"] = len(fig.data) - 1
    # axis locks
    fig.update_xaxes(range=[xmin, xmax], autorange=False, showgrid=True, zeroline=False)
    fig.update_xaxes(
        tickmode="linear", dtick="M3", tickformat="%Y-%m",
        showticklabels=True, automargin=True, row=4, col=1
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)

    # y ranges (hard lock)
    eq_all = pd.concat([df["bh_equity_norm"], df["strategy_equity_norm"]]).dropna()
    if not eq_all.empty:
        y1_min, y1_max = float(eq_all.min()), float(eq_all.max())
    else:
        y1_min, y1_max = 0.95, 1.05
    y1_pad = 0.02 * (y1_max - y1_min) if y1_max > y1_min else 0.02
    fig.update_yaxes(range=[y1_min - y1_pad, y1_max + y1_pad], autorange=False, row=1, col=1)

    if has_ret:
        r = pd.to_numeric(df[ret_col], errors="coerce")
        if r.notna().any():
            rmax = float(r.abs().max())
            rpad = 0.05 * rmax if rmax > 0 else 0.01
            fig.update_yaxes(range=[-(rmax + rpad), (rmax + rpad)], autorange=False, row=2, col=1)
        else:
            fig.update_yaxes(autorange=False, row=2, col=1)
    else:
        fig.update_yaxes(autorange=False, row=2, col=1)

    w = pd.to_numeric(df["weight"], errors="coerce")
    if w.notna().any():
        wmin, wmax = float(w.min()), float(w.max())
        wpad = 0.05 * (wmax - wmin) if wmax > wmin else 0.05
        fig.update_yaxes(range=[wmin - wpad, wmax + wpad], autorange=False, row=3, col=1)

    sv = pd.to_numeric(df["state_var"], errors="coerce")
    t_star = pd.to_numeric(df["tau_star"], errors="coerce")

    # combine both
    combined = pd.concat([sv, t_star]).dropna()

    if not combined.empty:
        vmin, vmax = float(combined.min()), float(combined.max())
        vpad = 0.05 * (vmax - vmin) if vmax > vmin else 0.01

        fig.update_yaxes(
            range=[vmin - vpad, vmax + vpad],
            autorange=False,
            row=4, col=1,
        )

    fig.update_layout(
        uirevision=f"run:{run_id}" if run_id else "lock",
        yaxis=dict(autorange=False, fixedrange=True),
        yaxis2=dict(autorange=False, fixedrange=True),
        yaxis3=dict(autorange=False, fixedrange=True),
        yaxis4=dict(autorange=False, fixedrange=True),
    )

    if lock_xticks:
        fig.update_xaxes(nticks=6)

    # optional regime shading
    _add_regime_shading(fig, df, xcol="session_date", signal_col="weight")

    # frames: update only the traces in the order they were added
    series = [
        ("line", "bh_equity_norm", "bh"),
        ("line", "strategy_equity_norm", "strategy"),
    ]
    if has_ret:
        series.append(("bar" if returns_as_bars else "line", ret_col, "ret"))
    series += [
        ("line", "weight", "weight"),
        ("line", "state_var", "state"),
    ]

    # Which traces to animate (NOT including "strategy")
    animate_keys = ["bh"] + (["ret"] if has_ret else []) + ["weight", "state", "tau"]
    animate_traces = [trace_idx[k] for k in animate_keys]

    FrameScatter =  go.Scatter

    frames = []
    timeline = []

    for i in range(0, len(df), every):
        sl = slice(0, i + 1)
        timeline.append(df["session_date"].iloc[i])

        frame_data = []

        # BH (animated)
        frame_data.append(FrameScatter(x=x.iloc[sl], y=df["bh_equity_norm"].iloc[sl], mode="lines"))

        # Returns (animated)
        if has_ret:
            r = pd.to_numeric(df[ret_col], errors="coerce").iloc[sl]
            if returns_as_bars:
                colors = np.where(r.fillna(0.0) >= 0, PAL["pos"], PAL["neg"]).tolist()

                frame_data.append(
                    go.Bar(
                        x=x.iloc[sl],
                        y=r,
                        marker_color=colors,   # <<< consistent
                    )
                )
            else:
                frame_data.append(FrameScatter(x=x.iloc[sl], y=r))

        # Weight (animated)
        frame_data.append(FrameScatter(x=x.iloc[sl], y=df["weight"].iloc[sl], mode="lines"))

        # State var (animated)
        frame_data.append(FrameScatter(x=x.iloc[sl], y=df["state_var"].iloc[sl], mode="lines"))

        # Tau* (animated)
        frame_data.append(FrameScatter(x=x.iloc[sl], y=df["tau_star"].fillna(0).iloc[sl], mode="lines"))

        frames.append(go.Frame(
            name=str(len(frames)),
            data=frame_data,
            traces=animate_traces,   # aligns 1:1 with frame_data order above
        ))

    fig.frames = frames


    # --- regime legend entries (add AFTER frames so trace indices don't shift) ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(34,197,94,0.25)"),
            name="Regime: Buy",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(239,68,68,0.25)"),
            name="Regime: No-Buy",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # controls
    fig.update_layout(
        template="plotly_white",
        height=940,
        margin=dict(l=20, r=20, t=90, b=40),
        legend=dict(
            orientation="h",
            y=1.02,
            yanchor="bottom",
            x=0.0,
            xanchor="left",
            font=dict(size=12),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.25,
                xanchor="left",
                yanchor="top",
                showactive=False,
                pad=dict(t=4, r=12),
                bgcolor="#1f2937",
                bordercolor="#111827",
                borderwidth=1,
                font=dict(size=14, color="white"),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.0,
                y=1.20,
                xanchor="left",
                yanchor="top",
                len=1.0,
                active=max(len(timeline) - 1, 0),
                pad=dict(t=10, b=0),
                currentvalue=dict(prefix="Date: ", font=dict(size=12)),
                steps=[
                    dict(
                        method="animate",
                        label=str(t)[:10],
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    )
                    for i, t in enumerate(timeline)
                ],
            )
        ],
    )

    fig.update_yaxes(title_text="Equity (Normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_yaxes(title_text="Weight", row=3, col=1)
    fig.update_yaxes(title_text="State Variable", row=4, col=1)
    fig.update_xaxes(title_text="Time", title_standoff=18, row=4, col=1)

    if debug:
        print("[plot_debug] frames:", len(timeline), "every:", every, "frame_ms:", frame_ms)

    return fig
