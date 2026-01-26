from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx

from services.data_access import (
    load_runs_with_metrics,
    read_timeseries_for_run,
    read_meta_for_run,
    meta_params_table,
    ResultsStore,
)

from layout.components import badge_row, kv_panel
from plots.plots import PLOT_REGISTRY
import plotly.express as px
import plotly.graph_objects as go


def _df_to_dash(df: pd.DataFrame):
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]


def _compute_kpis(ts: pd.DataFrame) -> dict:
    ts = ts.copy()
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts.sort_index()

    if "equity_net" in ts.columns:
        eq = ts["equity_net"].astype(float)
        ret = eq.pct_change()
    else:
        rc = "ret_net" if "ret_net" in ts.columns else ("ret" if "ret" in ts.columns else None)
        if rc is None:
            return dict(total_return=None, cagr=None, vol=None, mdd=None, sharpe=None)
        ret = ts[rc].astype(float)
        eq = (1.0 + ret.fillna(0.0)).cumprod()

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else None

    ann = 252.0
    r = ret.dropna()
    vol = float(r.std(ddof=0) * np.sqrt(ann)) if r.shape[0] > 2 else None
    sharpe = float((r.mean() / r.std(ddof=0)) * np.sqrt(ann)) if r.shape[0] > 2 and r.std(ddof=0) > 0 else None

    if len(eq) > 1 and pd.notna(ts.index[0]) and pd.notna(ts.index[-1]):
        years = (ts.index[-1] - ts.index[0]).days / 365.25
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else None
    else:
        cagr = None

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min()) if len(dd) else None

    return dict(total_return=total_return, cagr=cagr, vol=vol, mdd=mdd, sharpe=sharpe)


def register_callbacks(app, store_ctx: ResultsStore):

    @app.callback(
        # strategy dropdown
        Output("strategy_dd", "options"),
        Output("strategy_dd", "value"),

        # universe dropdown
        Output("universe_dd", "options"),
        Output("universe_dd", "value"),

        # run dropdown
        Output("run_dd", "options"),
        Output("run_dd", "value"),

        # ✅ NEW: compare dropdown
        Output("compare_run_dd", "options"),
        Output("compare_run_dd", "value"),

        # runs table
        Output("runs_table", "data"),
        Output("runs_table", "columns"),
        Output("runs_table", "tooltip_data"),

        # status
        Output("status", "children"),
        Output("status", "color"),

        # triggers
        Input("strategy_dd", "id"),
        Input("strategy_dd", "value"),
        Input("universe_dd", "value"),

        # ✅ NEW: keep compare selection if valid
        Input("compare_run_dd", "value"),

        Input("runs_table", "active_cell"),
        State("runs_table", "data"),
        Input("sort_metric_dd", "value"),
        Input("sort_dir", "value"),
    )
    def init_and_filter(_, strategy_value, universe_value, compare_value, active_cell, table_rows, sort_metric, sort_dir):
        runs = load_runs_with_metrics(store_ctx)
        if runs.empty:
            return (
                [], None,
                [], None,
                [], None,
                [], None,          # compare opts/value
                [], [],            # table data/cols
                [],                # tooltip_data
                "No runs found. Run: python scripts/run_backtest.py",
                "warning",
            )

        # ---------- strategy options ----------
        strategies = sorted(runs["strategy_name"].dropna().unique()) if "strategy_name" in runs.columns else []
        strat_opts = [{"label": s, "value": s} for s in strategies]
        strat_val = strategy_value if strategy_value in strategies else None

        # ---------- universe options (depend on strategy filter) ----------
        runs_for_universe = runs
        if strat_val and "strategy_name" in runs_for_universe.columns:
            runs_for_universe = runs_for_universe.loc[runs_for_universe["strategy_name"] == strat_val].copy()

        universes = sorted(runs_for_universe["universe"].dropna().unique()) if "universe" in runs_for_universe.columns else []
        uni_opts = [{"label": u, "value": u} for u in universes]
        uni_val = universe_value if universe_value in universes else None

        # ---------- filter runs by BOTH ----------
        runs_f = runs.copy()
        if strat_val and "strategy_name" in runs_f.columns:
            runs_f = runs_f.loc[runs_f["strategy_name"] == strat_val].copy()
        if uni_val and "universe" in runs_f.columns:
            runs_f = runs_f.loc[runs_f["universe"] == uni_val].copy()

        if runs_f.empty:
            return (
                strat_opts, strat_val,
                uni_opts, uni_val,
                [], None,
                [], None,          # compare
                [], [],
                [],
                "No runs after filter.",
                "warning",
            )

        # ---------- sort (force numeric if needed) ----------
        if sort_metric and sort_metric in runs_f.columns:
            # common: metrics loaded from CSV can be strings
            if sort_metric in ["sharpe", "cagr", "max_dd", "volatility"]:
                runs_f[sort_metric] = pd.to_numeric(runs_f[sort_metric], errors="coerce")

            ascending = (sort_dir == "asc")
            runs_f = runs_f.sort_values(sort_metric, ascending=ascending, na_position="last")

        # ---------- run dropdown ----------
        if "label" not in runs_f.columns:
            cols = [c for c in ["run_id", "strategy_name", "universe"] if c in runs_f.columns]
            runs_f["label"] = runs_f[cols].astype(str).agg(" | ".join, axis=1)

        run_opts = [{"label": r["label"], "value": str(r["run_id"])} for _, r in runs_f.iterrows()]
        default_run_id = run_opts[0]["value"]

        # ---------- pick run based on click ----------
        run_val = default_run_id
        if ctx.triggered_id == "runs_table" and active_cell and table_rows:
            try:
                clicked_row = active_cell["row"]
                candidate = table_rows[clicked_row].get("run_id")
                run_val = str(candidate) if candidate else default_run_id
            except Exception:
                run_val = default_run_id


        compare_opts = [o for o in run_opts if o["value"] != run_val]
        valid_compare = {o["value"] for o in compare_opts}
        compare_val = compare_value if compare_value in valid_compare else None

        # ---------- table ----------
        show_cols = [c for c in ["created_at_utc", "run_id", "strategy_name", "universe"] if c in runs_f.columns]
        tv = runs_f[show_cols].copy()

        if "created_at_utc" in tv.columns:
            tv["created_at_utc"] = pd.to_datetime(tv["created_at_utc"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

        table_data = tv.to_dict("records")
        table_cols = [{"name": c, "id": c} for c in tv.columns]
        tooltip_data = [{k: {"value": str(v), "type": "text"} for k, v in row.items()} for row in table_data]

        return (
            strat_opts, strat_val,
            uni_opts, uni_val,
            run_opts, run_val,
            compare_opts, compare_val,
            table_data, table_cols, tooltip_data,
            f"Loaded {len(run_opts)} run(s).",
            "secondary",
        )

    # ------------------------------------------------------------
    # Selected run: badges + params + KPI cards + equity fig (with optional compare)
    # ------------------------------------------------------------
    @app.callback(
        Output("run_badges", "children"),
        Output("params_kv", "children"),
        Output("kpi_sharpe", "children"),
        Output("kpi_cagr", "children"),
        Output("kpi_vol", "children"),
        Output("kpi_mdd", "children"),
        Output("equity_fig", "figure"),
        Input("run_dd", "value"),
        Input("compare_run_dd", "value"),   # ✅ NEW
    )
    def update_run_details(run_id: str, compare_run_id: str | None):

        # ---------- empty state ----------
        if not run_id:
            fig = px.line(title="No run selected")
            return "", "", "-", "-", "-", "-", fig

        # ---------- load primary timeseries ----------
        try:
            _row, ts = read_timeseries_for_run(store_ctx, run_id)
        except Exception as e:
            fig = px.line(title="Unable to load timeseries")
            badges = badge_row([("Run", run_id), ("Error", "timeseries load failed")])
            params_component = kv_panel([("timeseries_error", repr(e))])
            return badges, params_component, "-", "-", "-", "-", fig

        if not isinstance(ts.index, pd.DatetimeIndex):
            ts = ts.copy()
            ts.index = pd.to_datetime(ts.index, errors="coerce")
        ts = ts.sort_index()

        # ---------- load meta ----------
        meta_err = None
        try:
            meta = read_meta_for_run(store_ctx, run_id)
            if not isinstance(meta, dict):
                raise TypeError(f"meta must be dict, got {type(meta)}")
        except Exception as e:
            meta = {}
            meta_err = repr(e)

        # ---------- badges ----------
        strategy = meta.get("strategy_name") or (meta.get("spec") or {}).get("strategy_name") or ""
        universe = meta.get("universe") or (meta.get("spec") or {}).get("universe") or ""
        created = meta.get("created_at_utc") or meta.get("created_at") or ""

        badges_list = [("Run", run_id), ("Strategy", strategy), ("Universe", universe), ("Created", created)]
        if compare_run_id and compare_run_id != run_id:
            badges_list.append(("Compare", compare_run_id))
        if meta_err:
            badges_list.append(("Meta", "FAILED"))
        badges_comp = badge_row(badges_list)

        # ---------- params kv ----------
        try:
            params_df = meta_params_table(meta)
        except Exception as e:
            params_df = pd.DataFrame([{"parameter": "params_error", "value": repr(e)}])

        if meta_err:
            params_df = pd.concat(
                [pd.DataFrame([{"parameter": "meta_error", "value": meta_err}]), params_df],
                ignore_index=True,
            )

        if params_df.empty:
            items = [("info", "No parameters found")]
        else:
            if not {"parameter", "value"}.issubset(params_df.columns):
                cols = list(params_df.columns)
                if len(cols) >= 2:
                    params_df = params_df.rename(columns={cols[0]: "parameter", cols[1]: "value"})
                elif len(cols) == 1:
                    params_df = params_df.rename(columns={cols[0]: "value"})
                    params_df["parameter"] = "value"
                else:
                    params_df = pd.DataFrame([{"parameter": "info", "value": "No parameters found"}])

            items = [(str(r["parameter"]), str(r["value"])) for r in params_df[["parameter", "value"]].to_dict("records")]

        params_component = kv_panel(items)

        # ---------- KPIs (primary only) ----------
        k = _compute_kpis(ts)

        def fmt_pct(x):
            return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"

        def fmt_num(x):
            return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"

        kpi_sharpe = fmt_num(k.get("sharpe"))
        kpi_cagr = fmt_pct(k.get("cagr"))
        kpi_vol = fmt_pct(k.get("vol"))
        kpi_mdd = fmt_pct(k.get("mdd"))

        # ---------- equity plot (overlay compare) ----------
        # Prefer your plot registry if it exists, but overlay is easiest with go.Figure.
        def _equity_series(ts_df: pd.DataFrame) -> pd.Series | None:
            if "equity_net" in ts_df.columns:
                s = ts_df["equity_net"].astype(float)
                return s / s.iloc[0] if len(s) else s
            # fallback: build from returns
            rc = "ret_net" if "ret_net" in ts_df.columns else ("ret" if "ret" in ts_df.columns else None)
            if rc is None:
                return None
            r = ts_df[rc].astype(float).fillna(0.0)
            s = (1.0 + r).cumprod()
            return s / s.iloc[0] if len(s) else s

        eq1 = _equity_series(ts)
        fig = go.Figure()

        if eq1 is not None and len(eq1):
            fig.add_trace(go.Scatter(x=eq1.index, y=eq1.values, mode="lines", name=f"{strategy or 'Run'} | {universe or ''}".strip()))
        else:
            fig = px.line(title="Equity unavailable")

        if compare_run_id and compare_run_id != run_id:
            try:
                meta2 = read_meta_for_run(store_ctx, compare_run_id) or {}
                strat2 = meta2.get("strategy_name") or (meta2.get("spec") or {}).get("strategy_name") or compare_run_id
                uni2 = meta2.get("universe") or (meta2.get("spec") or {}).get("universe") or ""
                _row2, ts2 = read_timeseries_for_run(store_ctx, compare_run_id)
                if not isinstance(ts2.index, pd.DatetimeIndex):
                    ts2 = ts2.copy()
                    ts2.index = pd.to_datetime(ts2.index, errors="coerce")
                ts2 = ts2.sort_index()

                eq2 = _equity_series(ts2)
                if eq2 is not None and len(eq2):
                    fig.add_trace(
                        go.Scatter(
                            x=eq2.index,
                            y=eq2.values,
                            mode="lines",
                            name=f"{strat2} | {uni2}".strip(),
                            line=dict(dash="dash"),  
                        )
                    )
            except Exception:
                # silently ignore compare failures (keeps main run working)
                pass

        fig.update_layout(
            title="Equity curve (normalized)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        return badges_comp, params_component, kpi_sharpe, kpi_cagr, kpi_vol, kpi_mdd, fig
