import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import plotly.graph_objects as go


PAL = {
    "bh": "#64748b",        # slate (buy&hold)
    "strategy": "#0f172a",  # near-black navy
    "pos": "#16a34a",       # green
    "neg": "#dc2626",       # red
    "weight": "#7c3aed",    # violet
    "state": "#f59e0b",     # amber
    "tau": "#06b6d4",       # cyan
    "grid": "rgba(15,23,42,0.08)",
    "regime_buy": "rgba(100,100,100,0.08)",   # soft green band
}


def plot_rv_tau_weights_returns_equity_animated(
    dfr: pd.DataFrame,
    *,
    run_id: str | None = None,
    frame_ms: int = 200,
    every: int = 5,
    add_cursor: bool = False ,
    returns_as_bars: bool = True,   # bars stutter more when x-range animates
    use_webgl: bool = True,          # Scattergl smoother for long series
    lock_xticks: bool = True,        # reduces perceived "wobble" from tick recompute
    debug: bool = False,              # prints diagnostics
) -> go.Figure:
    # ---------------------------------------------------------------------
    # 0) Copy + basic timestamp cleaning
    # ---------------------------------------------------------------------
    df = dfr.copy()

    if "timestamp" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required column: timestamp",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No rows after timestamp cleaning.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    # ---------------------------------------------------------------------
    # 1) Debug: duplicates + dtypes + missingness
    # ---------------------------------------------------------------------
    dup_ts = int(df["timestamp"].duplicated().sum())
    if dup_ts > 0:
        df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")

    cols_to_check = [c for c in ["bh_equity", "equity_net", "ret", "weights", "state_var", "tau", "tau_star"] if c in df.columns]

    if debug:
        print("\n[animate_debug] rows:", len(df))
        print("[animate_debug] duplicate timestamps removed:", dup_ts)
        print("[animate_debug] dtypes:\n", df[["timestamp"] + cols_to_check].dtypes)
        if cols_to_check:
            print("[animate_debug] NA counts:\n", df[cols_to_check].isna().sum().to_string())

    # ---------------------------------------------------------------------
    # 2) Coerce numeric columns once (avoid per-trace to_numeric surprises)
    # ---------------------------------------------------------------------
    for c in cols_to_check:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional: forward-fill series where continuity is expected
    # (DO NOT ffill returns)
    for c in ["bh_equity", "equity_net", "weights", "state_var", "tau", "tau_star"]:
        if c in df.columns and c != "ret":
            df[c] = df[c].ffill()

    # ---------------------------------------------------------------------
    # 3) Validate required columns for this figure
    # ---------------------------------------------------------------------
    required = {"bh_equity", "equity_net"}
    missing = required - set(df.columns)
    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {', '.join(sorted(missing))}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    # ---------------------------------------------------------------------
    # 4) Figure scaffold
    # ---------------------------------------------------------------------
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.2, 0.18, 0.3],
        subplot_titles=("Equity", "Returns", "Weights / Exposure", "State Variable & Threshold"),
    )

    for a in fig.layout.annotations:
        a.update(
            font=dict(
                size=15,
                color="#111827",
                family="Arial Black",  # bold-looking font
            )
        )
    ScatterLine = go.Scattergl if use_webgl else go.Scatter

    x = df["timestamp"]
    xmin, xmax = x.min(), x.max()

    # ---------------------------------------------------------------------
    # 5) Add full traces ONCE
    # ---------------------------------------------------------------------
    bh = df["bh_equity"]
    eq = df["equity_net"]

    fig.add_trace(
        ScatterLine(
            x=x,
            y=bh,
            mode="lines",
            name="Buy & Hold",
            line=dict(dash="dash", color=PAL["bh"]),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        ScatterLine(
            x=x,
            y=eq,
            mode="lines",
            name="Strategy",
            line=dict(width=3, color=PAL["strategy"]),
        ),
        row=1, col=1,
    )

    has_ret = "ret" in df.columns
    r = df["ret"] if has_ret else None
    if has_ret:
        if returns_as_bars:
            colors = np.where(r >= 0, PAL["pos"], PAL["neg"])

            fig.add_trace(
                go.Bar(
                    x=x,
                    y=r,
                    name="Return",
                    marker=dict(color=colors),
                ),
                row=2, col=1,
            )
        else:
            fig.add_trace(ScatterLine(x=x, y=r, mode="lines", name="Return"), row=2, col=1)

    has_w = "weights" in df.columns
    w = df["weights"] if has_w else None
    if has_w:
        fig.add_trace(ScatterLine(x=x, y=w, mode="lines", name="Weight",
                                  line=dict(color=PAL["weight"], width=2)), row=3, col=1)

    has_rv = "state_var" in df.columns
    rv = df["state_var"] if has_rv else None
    if has_rv:
        fig.add_trace(
            ScatterLine(x=x, y=rv, mode="lines", name="State Var",
                        line=dict(color=PAL["state"], width=2)),
            row=4, col=1,
        )

        if "tau" in df.columns:
            tau = df["tau"]
            fig.add_trace(
                ScatterLine(x=x, y=tau, mode="lines", name="τ(t)", line=dict(dash="dash", color=PAL["tau"])),
                row=4, col=1,
            )
        elif "tau_star" in df.columns:
            tau_star = df["tau_star"]
            fig.add_trace(
                ScatterLine(x=x, y=tau_star, mode="lines", name="τ*", line=dict(dash="dash", color=PAL["tau"])),
                row=4, col=1,
            )

    # ---------------------------------------------------------------------
    # 6) Lock axis ranges (explicit ranges for all y-axes)
    # ---------------------------------------------------------------------
    # lock range on all x-axes
    fig.update_xaxes(range=[xmin, xmax], autorange=False)

    # styling on all x-axes (grid etc.)
    fig.update_xaxes(
        showgrid=True,
        zeroline=False,
        automargin=False,
    )

    # ONLY bottom subplot controls ticks/labels (prevents "weird labels" up top)
    fig.update_xaxes(
        tickmode="linear",
        dtick="M3",
        tickformat="%Y-%m",
        showticklabels=True,
        tickfont=dict(
            size=14,        # bigger
            color="black",
            family="Arial Black",  # bold look (Plotly doesn't have true font-weight)
        ),
        automargin=True,
        row=4, col=1,
    )
    # hide x tick labels on upper panels
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)

    # Equity y
    eq_all = pd.concat([bh, eq]).dropna()
    if not eq_all.empty:
        y1_min, y1_max = float(eq_all.min()), float(eq_all.max())
    else:
        y1_min, y1_max = 0.0, 1.0
    y1_pad = 0.02 * (y1_max - y1_min) if y1_max > y1_min else 1.0
    y1_lo, y1_hi = y1_min - y1_pad, y1_max + y1_pad
    fig.update_yaxes(range=[y1_lo, y1_hi], autorange=False, row=1, col=1)

    # Returns y
    if has_ret and r is not None and r.notna().any():
        rmax = float(r.abs().max())
        rpad = 0.05 * rmax if rmax > 0 else 0.01
        fig.update_yaxes(range=[-(rmax + rpad), (rmax + rpad)], autorange=False, row=2, col=1)
    else:
        fig.update_yaxes(autorange=False, row=2, col=1)

    # Weights y
    if has_w and w is not None and w.notna().any():
        wmin, wmax = float(w.min()), float(w.max())
        wpad = 0.05 * (wmax - wmin) if wmax > wmin else 0.05
        fig.update_yaxes(range=[wmin - wpad, wmax + wpad], autorange=False, row=3, col=1)
    else:
        fig.update_yaxes(autorange=False, row=3, col=1)

    # RV y
    if has_rv and rv is not None and rv.notna().any():
        vmin, vmax = float(rv.min()), float(rv.max())
        vpad = 0.05 * (vmax - vmin) if vmax > vmin else 0.01
        fig.update_yaxes(range=[vmin - vpad, vmax + vpad], autorange=False, row=4, col=1)
    else:
        fig.update_yaxes(autorange=False, row=4, col=1)

    # Hard lock all y-axes to prevent frame relayout autoscale
    fig.update_layout(
        uirevision=f"run:{run_id}" if run_id else "lock",
        yaxis=dict(autorange=False, fixedrange=True),
        yaxis2=dict(autorange=False, fixedrange=True),
        yaxis3=dict(autorange=False, fixedrange=True),
        yaxis4=dict(autorange=False, fixedrange=True),
    )

    if lock_xticks:
        fig.update_xaxes(nticks=6)


    # shapes_turnover = []
    # if "turnover" in df.columns:
    #     to = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0)
    #     eps = 1e-8
    #     event = to > eps

    #     has_signal = "signal" in df.columns
    #     sig = pd.to_numeric(df["signal"], errors="coerce") if has_signal else None

    #     if has_signal and sig is not None and sig.notna().any():
    #         sig_prev = sig.shift(1)
    #         flip_up = (sig_prev == 0) & (sig == 1)
    #         flip_dn = (sig_prev == 1) & (sig == 0)


    #         for ts in df.loc[flip_up, "timestamp"]:
    #             shapes_turnover.append(
    #                 dict(
    #                     type="line",
    #                     xref="x", yref="y",          # IMPORTANT: equity panel axes
    #                     x0=ts, x1=ts,
    #                     y0=y1_lo, y1=y1_hi,
    #                     opacity=0.20,
    #                     line=dict(width=1.5, color="#16a34a"),
    #                     layer="above",
    #                 )
    #             )
    #         for ts in df.loc[flip_dn, "timestamp"]:
    #             shapes_turnover.append(
    #                 dict(
    #                     type="line",
    #                     xref="x", yref="y",
    #                     x0=ts, x1=ts,
    #                     y0=y1_lo, y1=y1_hi,
    #                     opacity=0.20,
    #                     line=dict(width=1.5, color="#dc2626"),
    #                     layer="above",
    #                 )
    #             )
    #     else:
    #         for ts in df.loc[event, "timestamp"]:
    #             shapes_turnover.append(
    #                 dict(
    #                     type="line",
    #                     xref="x", yref="y",
    #                     x0=ts, x1=ts,
    #                     y0=y1_lo, y1=y1_hi,
    #                     opacity=0.15,
    #                     line=dict(width=1, color="blue"),
    #                     layer="above",
    #                 )
    #             )

    # # Attach shapes once (they'll get clipped by x-range during animation)
    # if shapes_turnover:
    #     fig.update_layout(shapes=list(fig.layout.shapes) + shapes_turnover if fig.layout.shapes else shapes_turnover)
   
    if "signal" in df.columns:
        sig = pd.to_numeric(df["signal"], errors="coerce")

        long_regions = []
        sell_regions = []

        start = None
        current_regime = None

        for ts, s in zip(df["timestamp"], sig):
            if s != current_regime:
                if start is not None:
                    if current_regime == 1:
                        long_regions.append((start, prev_ts))
                    elif current_regime == 0:
                        sell_regions.append((start, prev_ts))
                start = ts
                current_regime = s
            prev_ts = ts

        # close last regime
        if start is not None:
            if current_regime == 1:
                long_regions.append((start, df["timestamp"].iloc[-1]))
            elif current_regime == 0:
                sell_regions.append((start, df["timestamp"].iloc[-1]))

        shapes = []

        # Long regime shading (light green)
        for t0, t1 in long_regions:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=t0,
                    x1=t1,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(100,100,100,0.06)",  # soft green
                    line=dict(width=0),
                    layer="below",
                    name="regime_long",
                )
            )

        # Sell regime shading (light red)
        # for t0, t1 in sell_regions:
        #     shapes.append(
        #         dict(
        #             type="rect",
        #             xref="x",
        #             yref="paper",
        #             x0=t0,
        #             x1=t1,
        #             y0=0,
        #             y1=1,
        #             fillcolor="rgba(239,68,68,0.06)",  # soft red
        #             line=dict(width=0),
        #             layer="below",
        #             name="regime_sell",
        #         )
        #     )

        fig.update_layout(shapes=shapes)
   
    # ---------------------------------------------------------------------
    # 7) Cursor trace (use equity y-range, NOT [0,1])
    # ---------------------------------------------------------------------
    cursor_trace_idxs: list[int] = []
    if add_cursor:
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[0], x.iloc[0]],
                y=[y1_lo, y1_hi],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                name="Cursor",
            ),
            row=1, col=1,
        )
        cursor_trace_idxs.append(len(fig.data) - 1)

    # ---------------------------------------------------------------------
    # 8) Frames: update ONLY xaxis range (+ cursor x)
    # ---------------------------------------------------------------------
    timeline = x.iloc[::every].to_list()
    frames: list[go.Frame] = []
    for i, t in enumerate(timeline):
        frame_layout = go.Layout(xaxis=dict(range=[xmin, t], autorange=False))

        if add_cursor and cursor_trace_idxs:
            frame_data = [go.Scatter(x=[t, t], y=[y1_lo, y1_hi])]
            frames.append(go.Frame(name=str(i), layout=frame_layout, data=frame_data, traces=cursor_trace_idxs))
        else:
            frames.append(go.Frame(name=str(i), layout=frame_layout))

    fig.frames = frames

   
    # ---------------------------------------------------------------------
    # 9) Controls
    # ---------------------------------------------------------------------
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

                bgcolor="#1f2937",        # dark slate background
                bordercolor="#111827",
                borderwidth=1,
                font=dict(
                    size=14,              # larger text
                    color="white",        # text color
                ),

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
                y=1.20,              # sits neatly under buttons
                xanchor="left",
                yanchor="top",
                len=1.0,
                active=max(len(timeline) - 1, 0),
                pad=dict(t=10, b=0),
                currentvalue=dict(
                    prefix="Date: ",
                    font=dict(size=12),
                ),
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


    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_yaxes(title_text="Weight", row=3, col=1)
    fig.update_yaxes(title_text="State Variable/ τ", row=4, col=1)
    fig.update_xaxes(
        title_text="Time",
        title_standoff=18,   # pushes title downward
        row=4, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[x.iloc[0], x.iloc[0]],
            y=[y1_lo, y1_lo],  # irrelevant; won't be visible
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(100,100,100,0.10)",
            name="Buying Regime",
            visible=True,           # legend controls this trace visibility
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.01,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
            font=dict(size=12),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )
    if debug:
        print("[animate_debug] frames:", len(timeline), "every:", every, "frame_ms:", frame_ms)
        print("[animate_debug] returns_as_bars:", returns_as_bars, "use_webgl:", use_webgl, "cursor:", add_cursor)

    return fig
