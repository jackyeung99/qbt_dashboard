import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



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

