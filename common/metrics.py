from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def _to_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").dropna().astype(float)


def _equity_simple(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def _equity_log(r: pd.Series) -> pd.Series:
    return np.exp(r.cumsum())

def _sharpe(ret: pd.Series, ann_factor: int) -> float:
    mu = ret.mean()
    sd = ret.std(ddof=0)
    if sd <= 0:
        return np.nan
    return float((mu / sd) * np.sqrt(ann_factor))


def _cagr(equity: pd.Series, ann_factor: int) -> float:
    if len(equity) < 2:
        return np.nan
    # infer periods per year from ann_factor (assume daily)
    years = (len(equity) - 1) / ann_factor
    if years <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def _annual_vol(ret: pd.Series, ann_factor: int) -> float:
    return float(ret.std(ddof=0) * np.sqrt(ann_factor))



def _max_dd(equity: pd.Series):
    peak = equity.cummax()
    dd = equity / peak - 1.0

    return dd.min()

def _perf_metrics(
    r: pd.Series,
    *,
    ann_factor: int,
    return_type: str,
    prefix: str,
    initial_equity: float = 100_000,
) -> Dict[str, Any]:
    r = _to_series(r)
    if r.empty:
        return {f"{prefix}n_obs": 0}

    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    sharpe = (mean / vol) * np.sqrt(ann_factor) if vol > 0 else np.nan

    if return_type == "simple":
        eq = _equity_simple(r)
    elif return_type == "log":
        eq = _equity_log(r)
    else:
        raise ValueError(f"return_type must be 'simple' or 'log', got {return_type}")

    # scale equity to initial capital if desired
    eq = eq * float(initial_equity)

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())

    ending_equity = float(eq.iloc[-1])
    total_pnl = float(ending_equity - initial_equity)
    total_return = float(ending_equity / initial_equity - 1.0)

    cagr = (
        float((ending_equity / initial_equity) ** (ann_factor / len(r)) - 1.0)
        if len(r) > 0 and initial_equity > 0
        else np.nan
    )

    calmar = cagr / abs(max_dd) if np.isfinite(cagr) and max_dd < 0 else np.nan

    pos = r[r > 0]
    neg = r[r < 0]

    out = {
        f"{prefix}n_obs": int(len(r)),
        f"{prefix}mean": mean,
        f"{prefix}mean_ann": mean * ann_factor,
        f"{prefix}vol": vol,
        f"{prefix}vol_ann": vol * np.sqrt(ann_factor),
        f"{prefix}sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        f"{prefix}cagr": float(cagr) if np.isfinite(cagr) else None,
        f"{prefix}calmar": float(calmar) if np.isfinite(calmar) else None,
        f"{prefix}max_dd": max_dd,
        f"{prefix}avg_drawdown": float(dd.mean()),
        f"{prefix}ending_equity": ending_equity,
        f"{prefix}starting_equity": float(initial_equity),
        f"{prefix}total_pnl": total_pnl,
        f"{prefix}final_minus_initial_equity": total_pnl,
        f"{prefix}total_return": total_return,
        f"{prefix}best_period": float(r.max()),
        f"{prefix}worst_period": float(r.min()),
        f"{prefix}hit_rate": float((r > 0).mean()),
        f"{prefix}positive_periods": int((r > 0).sum()),
        f"{prefix}negative_periods": int((r < 0).sum()),
        f"{prefix}flat_periods": int((r == 0).sum()),
        f"{prefix}avg_gain": float(pos.mean()) if not pos.empty else None,
        f"{prefix}avg_loss": float(neg.mean()) if not neg.empty else None,
    }

    return out


def _run_lengths_bool(x: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a boolean series x, return (values, lengths) for consecutive runs.
    """
    if x.empty:
        return np.array([], dtype=bool), np.array([], dtype=int)

    v = x.to_numpy(dtype=bool)
    # boundaries where value changes
    change = np.flatnonzero(np.diff(v, prepend=v[0]))
    # compute run starts
    starts = np.r_[0, change]
    # run ends are next start, last end is len(v)
    ends = np.r_[starts[1:], len(v)]
    lengths = ends - starts
    values = v[starts]
    return values, lengths


def _signal_metrics(sig: pd.Series, *, ann_factor: int, prefix: str = "signal_") -> Dict[str, Any]:
    """
    Metrics for a 0/1 signal.

    Returns:
    - time spent in each state
    - transition counts
    - turnover frequency
    - run-length statistics
    - current regime / days in current regime
    """
    s = pd.to_numeric(sig, errors="coerce").fillna(0.0)
    s = (s > 0.5).astype(int)

    n = int(len(s))
    if n == 0:
        return {f"{prefix}n_obs": 0}

    prev = s.shift(1)
    changed = s.ne(prev)
    changed.iloc[0] = False

    n_flips = int(changed.sum())
    n_0_to_1 = int(((prev == 0) & (s == 1)).sum())
    n_1_to_0 = int(((prev == 1) & (s == 0)).sum())

    # run IDs for consecutive equal states
    run_id = s.ne(s.shift()).cumsum()

    run_sizes = s.groupby(run_id).size()
    run_state = s.groupby(run_id).first()

    lens_1 = run_sizes[run_state == 1].to_numpy()
    lens_0 = run_sizes[run_state == 0].to_numpy()
    lens_all = run_sizes.to_numpy()

    current_regime = int(s.iloc[-1])
    current_run_id = run_id.iloc[-1]
    days_in_current_regime = int(run_sizes.loc[current_run_id])

    previous_regime = None
    if len(run_state) >= 2:
        previous_regime = int(run_state.iloc[-2])

    out: Dict[str, Any] = {
        f"{prefix}n_obs": n,
        f"{prefix}pct_state_1": float(s.mean()),          # 0-1 fraction
        f"{prefix}pct_state_0": float(1.0 - s.mean()),   # 0-1 fraction
        f"{prefix}n_flips": n_flips,
        f"{prefix}n_turnovers": n_flips,
        f"{prefix}n_0_to_1": n_0_to_1,
        f"{prefix}n_1_to_0": n_1_to_0,
        f"{prefix}flips_per_year": float(n_flips) * (ann_factor / max(n, 1)),
        f"{prefix}turnovers_per_year": float(n_flips) * (ann_factor / max(n, 1)),
        f"{prefix}current_regime": current_regime,
        f"{prefix}previous_regime": previous_regime,
        f"{prefix}days_in_current_regime": days_in_current_regime,
        f"{prefix}avg_regime_duration": float(lens_all.mean()) if lens_all.size else None,
        f"{prefix}avg_hold_state_1": float(lens_1.mean()) if lens_1.size else None,
        f"{prefix}avg_hold_state_0": float(lens_0.mean()) if lens_0.size else None,
        f"{prefix}max_hold_state_1": int(lens_1.max()) if lens_1.size else None,
        f"{prefix}max_hold_state_0": int(lens_0.max()) if lens_0.size else None,
        f"{prefix}n_regimes": int(len(run_sizes)),
    }

    return out

def compute_portfolio_metrics(
    ts_df: pd.DataFrame,
    *,
    ann_factor: int = 252,
    return_type: str = "simple",
    col_ret: str = "strategy_ret",
    col_bh: Optional[str] = "bh_ret",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if col_ret not in ts_df.columns:
        raise KeyError(f"ts_df missing required column: {col_ret}")

    # strategy performance
    out.update(
        _perf_metrics(
            ts_df[col_ret],
            ann_factor=ann_factor,
            return_type=return_type,
            prefix="",
        )
    )

    # buy-and-hold benchmark
    if col_bh and col_bh in ts_df.columns:
        out.update(
            _perf_metrics(
                ts_df[col_bh],
                ann_factor=ann_factor,
                return_type=return_type,
                prefix="bh_",
            )
        )

        r_strategy = _to_series(ts_df[col_ret])
        r_bh = _to_series(ts_df[col_bh])
        idx = r_strategy.index.intersection(r_bh.index)

        if len(idx) > 2:
            out.update(
                _perf_metrics(
                    r_strategy.loc[idx] - r_bh.loc[idx],
                    ann_factor=ann_factor,
                    return_type=return_type,
                    prefix="excess_",
                )
            )

            strat_sh = out.get("sharpe")
            bh_sh = out.get("bh_sharpe")
            if (strat_sh is not None) and (bh_sh is not None):
                out["sharpe_minus_bh"] = float(strat_sh - bh_sh)

    # weight summaries
    weight_cols = [c for c in ts_df.columns if str(c).endswith("weight")]

    for col in weight_cols:
        w = pd.to_numeric(ts_df[col], errors="coerce").fillna(0.0)
        base = str(col)

        out[f"avg_{base}"] = float(w.mean())
        out[f"max_{base}"] = float(w.max())
        out[f"min_{base}"] = float(w.min())

    return out