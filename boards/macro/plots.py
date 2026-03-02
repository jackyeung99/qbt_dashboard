from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PAL = {
    "bh": "#0f172a",
    "strategy": "#dc2626",
    "pos": "#16a34a",
    "neg": "#dc2626",
    "weight": "#7c3aed",
    "state": "#f59e0b",
    "tau": "#06b6d4",
}


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _normalize_to_one(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.empty:
        return s
    s0 = s.dropna().iloc[0] if s.dropna().size else np.nan
    if not np.isfinite(s0) or s0 == 0:
        return pd.Series(np.nan, index=s.index)
    return s / float(s0)


def _add_regime_shading(fig: go.Figure, df: pd.DataFrame, *, xcol: str = "date", signal_col: str = "signal") -> None:
    """
    Shade buy/no-buy regimes based on signal in {0,1}.
    """
    if signal_col not in df.columns or xcol not in df.columns:
        return

    x = df[xcol]
    sig = pd.to_numeric(df[signal_col], errors="coerce").ffill()
    sig = sig.where(sig.isin([0, 1]))
    if not sig.notna().any() or df.empty:
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
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                x1=ts,
                y0=0, y1=1,
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            ))
            start = ts
            prev = s
        elif prev is None:
            start = ts
            prev = s

    if prev is not None:
        fill = "rgba(34,197,94,0.08)" if prev == 1 else "rgba(239,68,68,0.08)"
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=start,
            x1=x.iloc[-1],
            y0=0, y1=1,
            fillcolor=fill,
            line=dict(width=0),
            layer="below",
        ))

    fig.update_layout(shapes=shapes)


def plot_rv_tau_weights_returns_equity_animated(
    dfr: pd.DataFrame,
    *,
    run_id: str | None = None,
    frame_ms: int = 200,
    every: int = 1,
    returns_as_bars: bool = True,
    lock_xticks: bool = True,
    pal: Dict[str, str] | None = None,
    debug: bool = False,
) -> go.Figure:
    """
    Assumes df columns:
      date, equity_net, bh_equity, port_ret_net, weight_XLE, state_value, tau_star, signal
    """
    if pal:
        PAL.update(pal)

    if dfr is None or dfr.empty:
        return _empty_fig("No data.")

    df = dfr.copy()

  
    # --- enforce datetime + sort ---
    if "date" not in df.columns:
        return _empty_fig("Missing required column: date")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # --- required columns ---
    required = ["equity_net", "bh_equity", "port_ret_net", "weight_XLE", "state_value", "tau_star"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_fig(f"Missing required column(s): {', '.join(missing)}")

    # numeric coercion
    for c in required + ["signal"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- normalized equities ---
    df["eq_strat_norm"] = _normalize_to_one(df["equity_net"])
    df["eq_bh_norm"] = _normalize_to_one(df["bh_equity"])

    # forward-fill for smooth lines (NOT returns)
    df["weight_XLE"] = df["weight_XLE"].ffill()
    df["state_value"] = df["state_value"].ffill()
    df["tau_star"] = df["tau_star"].ffill()

    x = df["date"]
    xmin, xmax = x.min(), x.max()

    if debug:
        print("[plot] rows:", len(df), "xmin:", xmin, "xmax:", xmax)

    # --- figure scaffold (tau on secondary y-axis in row 4) ---
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.20, 0.18, 0.30],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=(
            "XLE Regime State Strategy",
            "XLE Return",
            "Weight (XLE)",
            "State Value and threshold τ*",
        ),
    )

    # --- initial slice for animation ---
    sl0 = slice(0, len(df))

    # row 1: equity lines (animated)
    fig.add_trace(
        go.Scatter(
            x=x.iloc[sl0], y=df["eq_bh_norm"].iloc[sl0],
            mode="lines",
            name="Buy & Hold (normalized)",
            line=dict(dash="dash", color=PAL["bh"], width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x.iloc[sl0], y=df["eq_strat_norm"].iloc[sl0],
            mode="lines",
            name="Strategy (normalized)",
            line=dict(color=PAL["strategy"], width=3),
        ),
        row=1, col=1,
    )

    # row 2: returns (animated bars or line)
    if returns_as_bars:
        r0 = df["bh_ret"].iloc[sl0].fillna(0.0)
        colors0 = np.where(r0 >= 0, PAL["pos"], PAL["neg"]).tolist()
        fig.add_trace(
            go.Bar(
                x=x.iloc[sl0],
                y=r0,
                name="Return (net)",
                marker_color=colors0,
                opacity=0.85,
            ),
            row=2, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x.iloc[sl0], y=df["bh_ret"].iloc[sl0],
                mode="lines",
                name="Return (net)",
                line=dict(width=2),
            ),
            row=2, col=1,
        )

    # row 3: weight (animated)
    fig.add_trace(
        go.Scatter(
            x=x.iloc[sl0], y=df["weight_XLE"].iloc[sl0],
            mode="lines",
            name="Weight_XLE",
            line=dict(color=PAL["weight"], width=2),
        ),
        row=3, col=1,
    )

    # row 4: state (left) + tau (right) (animated)
    fig.add_trace(
        go.Scatter(
            x=x.iloc[sl0], y=df["state_value"].iloc[sl0],
            mode="lines",
            name="State value",
            line=dict(color=PAL["state"], width=2),
        ),
        row=4, col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x.iloc[sl0], y=df["tau_star"].iloc[sl0],
            mode="lines",
            name="τ*",
            line=dict(color=PAL["tau"], width=2, dash="dot"),
        ),
        row=4, col=1,
        secondary_y=False,
    )

    # --- axis formatting ---
    fig.update_xaxes(range=[xmin, xmax], autorange=False, showgrid=True, zeroline=False)
    fig.update_xaxes(
        tickmode="linear", dtick="M3", tickformat="%Y-%m",
        showticklabels=True, automargin=True, row=4, col=1
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)

    # y-ranges (stable)
    eq_all = pd.concat([df["eq_bh_norm"], df["eq_strat_norm"]]).dropna()
    if not eq_all.empty:
        y1_min, y1_max = float(eq_all.min()), float(eq_all.max())
        pad = 0.02 * (y1_max - y1_min) if y1_max > y1_min else 0.02
        fig.update_yaxes(range=[y1_min - pad, y1_max + pad], autorange=False, row=1, col=1)

    r = df["port_ret_net"]
    if r.notna().any():
        rmax = float(r.abs().max())
        pad = 0.05 * rmax if rmax > 0 else 0.01
        fig.update_yaxes(range=[-(rmax + pad), (rmax + pad)], autorange=False, row=2, col=1)

    w = df["weight_XLE"]
    if w.notna().any():
        wmin, wmax = float(w.min()), float(w.max())
        pad = 0.05 * (wmax - wmin) if wmax > wmin else 0.05
        fig.update_yaxes(range=[wmin - pad, wmax + pad], autorange=False, row=3, col=1)

    sv = df["state_value"]
    tv = df["tau_star"]
    combo = pd.concat([sv, tv]).dropna()

    if combo.notna().any():
        vmin, vmax = float(combo.min()), float(combo.max())
        pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.01
        fig.update_yaxes(
            range=[vmin - pad, vmax + pad],
            autorange=False,
            row=4, col=1,
            secondary_y=False,
        ) 

    # --- regime shading (optional) ---
    _add_regime_shading(fig, df, xcol="date", signal_col="signal")

    # --- animation frames ---
    # trace order:
    # 0 bh, 1 strat, 2 ret, 3 weight, 4 state, 5 tau
    frames = []
    timeline: List[pd.Timestamp] = []

    for i in range(0, len(df), every):
        sl = slice(0, i + 1)
        timeline.append(df["date"].iloc[i])

        frame_data = [
            go.Scatter(x=x.iloc[sl], y=df["eq_bh_norm"].iloc[sl], mode="lines"),
            go.Scatter(x=x.iloc[sl], y=df["eq_strat_norm"].iloc[sl], mode="lines"),
        ]

        if returns_as_bars:
            rr = df["port_ret_net"].iloc[sl].fillna(0.0)
            colors = np.where(rr >= 0, PAL["pos"], PAL["neg"]).tolist()
            frame_data.append(go.Bar(x=x.iloc[sl], y=rr, marker_color=colors, opacity=0.85))
        else:
            frame_data.append(go.Scatter(x=x.iloc[sl], y=df["port_ret_net"].iloc[sl], mode="lines"))

        frame_data += [
            go.Scatter(x=x.iloc[sl], y=df["weight_XLE"].iloc[sl], mode="lines"),
            go.Scatter(x=x.iloc[sl], y=df["state_value"].iloc[sl], mode="lines"),
            go.Scatter(x=x.iloc[sl], y=df["tau_star"].iloc[sl], mode="lines"),
        ]

        frames.append(go.Frame(name=str(len(frames)), data=frame_data, traces=[0, 1, 2, 3, 4, 5]))

    fig.frames = frames

    # --- add regime legend markers WITHOUT shifting animated traces ---
    # (Add them as annotations instead of traces to avoid trace index changes)
    fig.add_annotation(
        xref="paper", yref="paper", x=1.0, y=1.05,
        text="<span style='color:rgba(34,197,94,0.55)'>■</span> Low Regime (State Variable <= τ*) &nbsp;&nbsp; "
             "<span style='color:rgba(239,68,68,0.55)'>■</span> High Regime (State Variable > τ*)",
        showarrow=False,
        align="right",
    )

    # --- controls ---
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
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                         transition=dict(duration=0),
                                         fromcurrent=True,
                                         mode="immediate")],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           transition=dict(duration=0),
                                           mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.0,
                y=1.2,
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
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0),
                                             mode="immediate")],
                    )
                    for i, t in enumerate(timeline)
                ],
            )
        ],
    )

    fig.update_yaxes(title_text="Equity (norm)", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_yaxes(title_text="Weight", row=3, col=1)
    fig.update_yaxes(title_text="State value and τ*", row=4, col=1, secondary_y=False)
    fig.update_xaxes(title_text="Time", title_standoff=18, row=4, col=1)

    if lock_xticks:
        fig.update_xaxes(nticks=6)

    return fig