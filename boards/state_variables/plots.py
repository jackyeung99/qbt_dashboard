import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Panel builders
# =============================================================================

def build_equity_panel(dfr: pd.DataFrame):
    g = (
        dfr[["timestamp", "bh_equity"]]
        .rename(columns={"bh_equity": "equity"})
        .copy()
    )

    s = (
        dfr[["timestamp", "equity_net", "turnover"]]
        .rename(columns={"equity_net": "equity"})
        .copy()
    )

    has_signal = "signal" in dfr.columns
    if has_signal:
        s["signal"] = pd.to_numeric(dfr["signal"], errors="coerce")

    # -----------------------------
    # Base traces (first point only)
    # -----------------------------
    traces = [
        go.Scatter(
            x=[g["timestamp"].iloc[0]] if len(g) else [],
            y=[g["equity"].iloc[0]] if len(g) else [],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="gray", dash="dash"),
        ),
        go.Scatter(
            x=[s["timestamp"].iloc[0]] if len(s) else [],
            y=[s["equity"].iloc[0]] if len(s) else [],
            mode="lines",
            name="Strategy",
            line=dict(color="black", width=3),
        ),
    ]

    # -----------------------------
    # Turnover shapes (precompute)
    # -----------------------------
    shapes_all = []

    if len(s):
        eps = 1e-8
        turnover_event = s["turnover"].fillna(0).astype(float) > eps

        if has_signal and s["signal"].notna().any():
            sig_prev = s["signal"].shift(1)

            flip_up = turnover_event & (sig_prev == 0) & (s["signal"] == 1)
            flip_dn = turnover_event & (sig_prev == 1) & (s["signal"] == 0)

            for ts in s.loc[flip_up, "timestamp"]:
                shapes_all.append(
                    dict(
                        type="line",
                        opacity=0.2,
                        x0=ts,
                        x1=ts,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(width=2, color="green"),
                    )
                )

            for ts in s.loc[flip_dn, "timestamp"]:
                shapes_all.append(
                    dict(
                        type="line",
                        opacity=0.2,
                        x0=ts,
                        x1=ts,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(width=2, color="red"),
                    )
                )

        else:
            for ts in s.loc[turnover_event, "timestamp"]:
                shapes_all.append(
                    dict(
                        type="line",
                        opacity=0.15,
                        x0=ts,
                        x1=ts,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(width=2, color="blue"),
                    )
                )

    # -----------------------------
    # Per-frame updates
    # -----------------------------
    def frame_at(t):
        gg = g[g["timestamp"] <= t]
        ss = s[s["timestamp"] <= t]

        return [
            go.Scatter(x=gg["timestamp"], y=gg["equity"]),
            go.Scatter(x=ss["timestamp"], y=ss["equity"]),
        ]

    def layout_at(t):
        return {}
    

    return traces, frame_at, layout_at


def build_ret_panel(dfr: pd.DataFrame):
    if "weights" not in dfr.columns:
        traces = [
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Return",
            )
        ]

        def frame_at(t):
            return [go.Scatter(x=[], y=[])]

        def layout_at(t):
            return {}

        return traces, frame_at, layout_at

    x = dfr["timestamp"]
    y = pd.to_numeric(dfr["weights"], errors="coerce")

    traces = [
        go.Scatter(
            x=[x.iloc[0]],
            y=[y.iloc[0]],
            mode="lines",
            name="Return",
        )
    ]

    def frame_at(t):
        mask = dfr["timestamp"] <= t
        return [go.Scatter(x=x[mask], y=y[mask])]

    def layout_at(t):
        return {}

    return traces, frame_at, layout_at


# =============================================================================
# Animator
# =============================================================================

def animate_together(df: pd.DataFrame) -> go.Figure:
    # ---------------------------------------------------------------------
    # 0) Start from the provided dataframe (assumed already filtered to run)
    # ---------------------------------------------------------------------
    dfr = df.copy()
    

    # ---------------------------------------------------------------------
    # 1) Validate required columns
    # ---------------------------------------------------------------------
    required = {"timestamp", "bh_equity", "equity_net", "turnover"}
    missing = required - set(dfr.columns)

    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing columns: {', '.join(sorted(missing))}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    # ---------------------------------------------------------------------
    # 2) Clean + sort timestamps (do this once, shared by all panels)
    # ---------------------------------------------------------------------
    dfr["timestamp"] = pd.to_datetime(dfr["timestamp"], errors="coerce")
    dfr = dfr.dropna(subset=["timestamp"]).sort_values("timestamp")

    if dfr.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No rows after timestamp cleaning.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    # ---------------------------------------------------------------------
    # 3) Shared timeline for animation (unique timestamps, in order)
    # ---------------------------------------------------------------------
    timeline = dfr["timestamp"].dropna().unique()

    # ---------------------------------------------------------------------
    # 4) Build panels
    # ---------------------------------------------------------------------
    eq_traces, eq_frame_at, eq_layout_at = build_equity_panel(dfr)
    ret_traces, ret_frame_at, ret_layout_at = build_ret_panel(dfr)

    # ---------------------------------------------------------------------
    # 5) Create a single figure with subplots so everything animates together
    # ---------------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )
    
    # ---------------------------------------------------------------------
    # 6) Add traces in a fixed order (this order MUST match frame updates)
    # ---------------------------------------------------------------------
    for tr in eq_traces:
        fig.add_trace(tr, row=1, col=1)

    for tr in ret_traces:
        fig.add_trace(tr, row=2, col=1)

    # ----- Equity Y bounds (row 1) -----
    equity_all = pd.concat(
        [
            dfr["bh_equity"].astype(float),
            dfr["equity_net"].astype(float),
        ],
        axis=0,
    )

    ymin = float(equity_all.min())
    ymax = float(equity_all.max())

    ypad = 0.02 * (ymax - ymin) if ymax > ymin else 1.0

    fig.update_yaxes(
        range=[ymin - ypad, ymax + ypad],
        autorange=False,
        row=1,
        col=1,
    )

    # ----- Return Y bounds (row 2) -----
    if "ret" in dfr.columns:
        ret_all = pd.to_numeric(dfr["ret"], errors="coerce")

        if ret_all.notna().any():
            rmax = float(ret_all.abs().max())
        else:
            rmax = 1.0

        rpad = 0.05 * rmax if rmax > 0 else 0.01

        fig.update_yaxes(
            range=[-(rmax + rpad), (rmax + rpad)],
            autorange=False,
            row=2,
            col=1,
        )

    # ---------------------------------------------------------------------
    # 8) Build frames (trace updates must match add_trace order)
    # ---------------------------------------------------------------------
    xmin = dfr["timestamp"].min()
    xmax_full = dfr["timestamp"].max()

    frames = []

    fig.update_xaxes(range=[xmin, xmax_full], autorange=False)

    SHAPE_EVERY = 10  # try 10, 20, 50
    for i, t in enumerate(timeline):
        data_updates = []
        data_updates += eq_frame_at(t)
        data_updates += ret_frame_at(t)

        layout_updates = {}


        frames.append(go.Frame(name=str(i), data=data_updates, layout=go.Layout(**layout_updates)))

    fig.frames = frames

    # ---------------------------------------------------------------------
    # 9) Animation controls
    # ---------------------------------------------------------------------
    frame_ms = 40
    transition_ms = 0

    fig.update_layout(
        height=650,  # more vertical room overall
        margin=dict(l=10, r=10, t=110, b=50),  # BIGGER TOP MARGIN for controls/slider
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.22,          # move buttons higher
                xanchor="left",
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=False),
                                transition=dict(duration=transition_ms),
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
                y=1.12,          # move slider higher
                xanchor="left",
                yanchor="top",
                len=1.0,
                pad=dict(t=0, b=0),
                steps=[
                    dict(
                        method="animate",
                        label=str(t)[:19],
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

    # ---------------------------------------------------------------------
    # 10) Labels
    # ---------------------------------------------------------------------
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)




import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_rv_tau_weights_returns_equity(dfr: pd.DataFrame) -> go.Figure:
    df = dfr.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=("Equity", "Returns", "Weights / Exposure", "Realized Variance & Threshold"),
    )

    # -----------------------
    # Row 1: Equity
    # -----------------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=pd.to_numeric(df["bh_equity"], errors="coerce"),
            mode="lines",
            name="Buy & Hold",
            line=dict(dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=pd.to_numeric(df["equity_net"], errors="coerce"),
            mode="lines",
            name="Strategy",
            line=dict(width=3),
        ),
        row=1, col=1,
    )

    # -----------------------
    # Row 2: Returns (bars)
    # -----------------------
    if "ret" in df.columns:
        r = pd.to_numeric(df["ret"], errors="coerce")
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=r,
                name="Return",
            ),
            row=2, col=1,
        )

    # -----------------------
    # Row 3: Weights
    # -----------------------
    if "weights" in df.columns:
        w = pd.to_numeric(df["weights"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=w,
                mode="lines",
                name="Weight",
            ),
            row=3, col=1,
        )

    # -----------------------
    # Row 4: RV + Threshold
    # -----------------------
    if "state_var" in df.columns:
        rv = pd.to_numeric(df["state_var"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=rv,
                mode="lines",
                name="RV",
            ),
            row=4, col=1,
        )

        # threshold: either a constant tau_star or a time series tau
        if "tau_star" in df.columns:
            tau = pd.to_numeric(df["tau_star"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=tau,
                    mode="lines",
                    name="τ(t)",
                    line=dict(dash="dash"),
                ),
                row=4, col=1,
            )
        elif "tau_star" in df.columns:
            tau_star = float(pd.to_numeric(df["tau_star"], errors="coerce").dropna().iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=[tau_star] * len(df),
                    mode="lines",
                    name="τ*",
                    line=dict(dash="dash"),
                ),
                row=4, col=1,
            )

    # -----------------------
    # Layout polish
    # -----------------------
    xmin, xmax = df["timestamp"].min(), df["timestamp"].max()
    fig.update_xaxes(range=[xmin, xmax], autorange=False)

    fig.update_layout(
        height=900,
        margin=dict(l=10, r=10, t=60, b=30),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_yaxes(title_text="Weight", row=3, col=1)
    fig.update_yaxes(title_text="RV / τ", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    return fig


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_rv_tau_weights_returns_equity_animated(
    dfr: pd.DataFrame,
    *,
    run_id: str | None = None,
    frame_ms: int = 200,
    every: int = 5,
    add_cursor: bool = False ,
    returns_as_bars: bool = False,   # bars stutter more when x-range animates
    use_webgl: bool = True,          # Scattergl smoother for long series
    lock_xticks: bool = True,        # reduces perceived "wobble" from tick recompute
    debug: bool = True,              # prints diagnostics
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
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=("Equity", "Returns", "Weights / Exposure", "Realized Variance & Threshold"),
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
            line=dict(dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        ScatterLine(
            x=x,
            y=eq,
            mode="lines",
            name="Strategy",
            line=dict(width=3),
        ),
        row=1, col=1,
    )

    has_ret = "ret" in df.columns
    r = df["ret"] if has_ret else None
    if has_ret:
        if returns_as_bars:
            fig.add_trace(go.Bar(x=x, y=r, name="Return"), row=2, col=1)
        else:
            fig.add_trace(ScatterLine(x=x, y=r, mode="lines", name="Return"), row=2, col=1)

    has_w = "weights" in df.columns
    w = df["weights"] if has_w else None
    if has_w:
        fig.add_trace(ScatterLine(x=x, y=w, mode="lines", name="Weight"), row=3, col=1)

    has_rv = "state_var" in df.columns
    rv = df["state_var"] if has_rv else None
    if has_rv:
        fig.add_trace(ScatterLine(x=x, y=rv, mode="lines", name="RV"), row=4, col=1)

        if "tau" in df.columns:
            tau = df["tau"]
            fig.add_trace(
                ScatterLine(x=x, y=tau, mode="lines", name="τ(t)", line=dict(dash="dash")),
                row=4, col=1,
            )
        elif "tau_star" in df.columns:
            tau_star = df["tau_star"]
            fig.add_trace(
                ScatterLine(x=x, y=tau_star, mode="lines", name="τ*", line=dict(dash="dash")),
                row=4, col=1,
            )

    # ---------------------------------------------------------------------
    # 6) Lock axis ranges (explicit ranges for all y-axes)
    # ---------------------------------------------------------------------
    fig.update_xaxes(range=[xmin, xmax], autorange=False)

    fig.update_xaxes(
        tickmode="linear",
        dtick="M3",              # every 3 months (try "M1" or "M6")
        tickformat="%Y-%m",      # stable label size
        showticklabels=True,
        automargin=False,
    )

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
    fig.update_xaxes(range=[xmin, xmax], autorange=False)
    frames: list[go.Frame] = []
    for i, t in enumerate(timeline):
        frame_layout = go.Layout(xaxis=dict(range=[xmin, t], autorange=False))

        if add_cursor and cursor_trace_idxs:
            frame_data = [go.Scatter(x=[t, t], y=[y1_lo, y1_hi])]
            frames.append(go.Frame(name=str(i), layout=frame_layout, data=frame_data, traces=cursor_trace_idxs))
        else:
            frames.append(go.Frame(name=str(i), layout=frame_layout))

    fig.frames = frames

    if len(timeline) > 5:
        fig.update_xaxes(range=[xmin, timeline[5]], autorange=False)

    # ---------------------------------------------------------------------
    # 9) Controls
    # ---------------------------------------------------------------------
    fig.update_layout(
        height=900,
        margin=dict(l=10, r=10, t=115, b=30),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.22,
                xanchor="left",
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=False),
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
                y=1.12,
                xanchor="left",
                yanchor="top",
                len=1.0,
                pad=dict(t=0, b=0),
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
    fig.update_yaxes(title_text="RV / τ", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    if debug:
        print("[animate_debug] frames:", len(timeline), "every:", every, "frame_ms:", frame_ms)
        print("[animate_debug] returns_as_bars:", returns_as_bars, "use_webgl:", use_webgl, "cursor:", add_cursor)

    return fig
