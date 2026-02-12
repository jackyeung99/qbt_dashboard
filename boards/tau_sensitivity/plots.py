import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from .queries import load_equity_all


def plot_equity_animated(run_id: str, strategy: str) -> go.Figure:
    df = load_equity_all()

    # --- slice buy&hold + strategy ---
    g = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == f"bh_{strategy}")].copy()
    s = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == strategy)].copy()

    all_eq = pd.concat(
    [
        g[["timestamp", "equity"]] if not g.empty else pd.DataFrame(),
        s[["timestamp", "equity"]] if not s.empty else pd.DataFrame(),
    ]
    )

    xmin = all_eq["timestamp"].min()
    xmax = all_eq["timestamp"].max()
    ymin = all_eq["equity"].min()
    ymax = all_eq["equity"].max()

    # small padding so lines don't hug edges
    ypad = 0.02 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    fig = go.Figure()

    if g.empty and s.empty:
        fig.add_annotation(text="No equity data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    # sort + unique timeline (use strategy timeline as driver if available)
    if not s.empty:
        s = s.sort_values("timestamp")
        timeline = s["timestamp"].dropna().unique()
    else:
        g = g.sort_values("timestamp")
        timeline = g["timestamp"].dropna().unique()

    # ---- base traces (start with first point only) ----
    # Buy & Hold (gray dashed)
    if not g.empty:
        g = g.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=[g["timestamp"].iloc[0]],
                y=[g["equity"].iloc[0]],
                mode="lines",
                name="Buy & Hold",
                line=dict(color="gray", dash="dash"),
            )
        )
    else:
        # placeholder so trace indices stay consistent
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 name="Buy & Hold",
                                 line=dict(color="gray", dash="dash")))

    # Strategy (black)
    if not s.empty and strategy != "bh":
        fig.add_trace(
            go.Scatter(
                x=[s["timestamp"].iloc[0]],
                y=[s["equity"].iloc[0]],
                mode="lines",
                name=strategy.upper(),
                line=dict(color="black", width=3),
            )
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 name=strategy.upper(),
                                 line=dict(color="black", width=3)))

    # ---- turnover flip logic (precompute) ----
    shapes_all = []
    if not s.empty and strategy != "bh" and {"signal_t", "turnover"}.issubset(s.columns):
        s2 = s.copy()
        sig_prev = s2["signal_t"].shift(1)

        flip_up = (sig_prev == 0) & (s2["signal_t"] == 1)
        flip_dn = (sig_prev == 1) & (s2["signal_t"] == 0)

        eps = 1e-8
        flip_up &= (s2["turnover"].fillna(0) > eps)
        flip_dn &= (s2["turnover"].fillna(0) > eps)

        for ts in s2.loc[flip_up, "timestamp"]:
            shapes_all.append(
                dict(
                    type="line",
                    opacity=0.2,
                    x0=ts, x1=ts,
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(width=2, color="green"),
                )
            )
        for ts in s2.loc[flip_dn, "timestamp"]:
            shapes_all.append(
                dict(
                    type="line",
                    opacity=0.2,
                    x0=ts, x1=ts,
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(width=2, color="red"),
                )
            )

    # helper: shapes revealed up to current timestamp
    def shapes_upto(t):
        if not shapes_all:
            return []
        out = []
        for sh in shapes_all:
            # x0 is the event day
            if sh["x0"] <= t:
                out.append(sh)
        return out

    # ---- build frames: reveal data up to each day ----
    frames = []
    for i, t in enumerate(timeline):
        # buy&hold partial
        if not g.empty:
            gg = g[g["timestamp"] <= t]
            xg, yg = gg["timestamp"], gg["equity"]
        else:
            xg, yg = [], []

        # strategy partial
        if not s.empty and strategy != "bh":
            ss = s[s["timestamp"] <= t]
            xs, ys = ss["timestamp"], ss["equity"]
        else:
            xs, ys = [], []

        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter(x=xg, y=yg),  # trace 0
                    go.Scatter(x=xs, y=ys),  # trace 1
                ],
                layout=go.Layout(shapes=shapes_upto(t)),
            )
        )

    fig.frames = frames

    # ---- animation controls ----
    # step per day; tune frame duration to taste
    frame_ms = 40
    transition_ms = 0

    fig.update_layout(
        xaxis=dict(range=[xmin, xmax], autorange=False),
        yaxis=dict(range=[ymin, ymax], autorange=False),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.15,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=True),
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
                y=1.08,
                len=1.0,
                pad=dict(t=10, b=0),
                steps=[
                    dict(
                        method="animate",
                        label=str(t)[:10],  # e.g. '2025-01-15'
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    )
                    for i, t in enumerate(timeline)
                ],
            )
        ],
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Equity",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )

    return fig


def plot_equity(run_id: str, strategy: str) -> go.Figure:
    df = load_equity_all()
    fig = go.Figure()



    g = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == f"bh_{strategy}")]
    if not g.empty:
        fig.add_trace(go.Scatter(x=g["timestamp"], y=g["equity"], mode="lines",
                                    name="Buy & Hold", line=dict(color="gray", dash="dash")))

    s = df[(df["run_id"] == run_id) & (df["strategy"].astype(str) == strategy)]
    if not s.empty and strategy != "bh":
        fig.add_trace(go.Scatter(x=s["timestamp"], y=s["equity"], mode="lines",
                                    name=strategy.upper(), line=dict(color= 'black', width=3)))
        
        # --- FULL-HEIGHT TURNOVER BARS ---
        shapes = []
        s2 = s.sort_values("timestamp").copy()
        sig_prev = s2["signal_t"].shift(1)

        # flip events
        flip_up = (sig_prev == 0) & (s2["signal_t"] == 1)
        flip_dn = (sig_prev == 1) & (s2["signal_t"] == 0)

        # (optional) also require meaningful turnover to avoid noise
        eps = 1e-8
        flip_up &= (s2["turnover"].fillna(0) > eps)
        flip_dn &= (s2["turnover"].fillna(0) > eps)

        for ts in s2.loc[flip_up, "timestamp"]:
            shapes.append(
                dict(
                    type="line",
                    opacity=0.2,
                    x0=ts, x1=ts,
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(width=2, color="green"),
                    
                )
            )

        for ts in s2.loc[flip_dn, "timestamp"]:
            shapes.append(
                dict(
                    type="line",
                    opacity=0.2,
                    x0=ts, x1=ts,
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(width=2, color="red"),
                )
            )
        fig.update_layout(shapes=shapes)

    if len(fig.data) == 0:
        fig.add_annotation(text="No equity data", xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False)

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Equity",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return fig



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

