from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

import pandas as pd


# ----------------------------
# Utils
# ----------------------------

def _clean(x: str) -> str:
    """Filesystem-safe string."""
    return str(x).strip().replace("/", "_").replace("\\", "_")


def _read_table(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """
    Prefer CSV if present (deployment-friendly). Fall back to parquet.
    Return empty df if neither exists.
    """
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def _read_timeseries(csv_path: Path, parquet_path: Path, time_col: str = "timestamp") -> pd.DataFrame:
    """
    Read timeseries and set time_col as DatetimeIndex.
    Works for either CSV (parse_dates) or parquet.
    """
    if csv_path.exists():
        # If time_col not present, parse_dates silently does nothing only if column exists,
        # but pandas will error if it's missing. So read first, then parse if present.
        df = pd.read_csv(csv_path)
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        return pd.DataFrame()

    if time_col in df.columns:
        df = df.set_index(time_col)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    return df


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict into dot keys: {'a': {'b': 1}} -> {'a.b': 1}"""
    out: dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


# ----------------------------
# Store
# ----------------------------

@dataclass(frozen=True)
class ResultsStore:
    """
    Read-only access to exported dashboard artifacts under <root>/results.

    Layout:
      results/
        runs.(csv|parquet)
        metrics.(csv|parquet)
        runs/<run_id>/meta.json
        timeseries/strategy=<strategy>/universe=<universe>/run_id=<run_id>/timeseries.(csv|parquet)
    """
    root: Path
    results_root: str = "results"

    @property
    def results_dir(self) -> Path:
        return self.root / self.results_root

    # ---- global artifacts ----
    @property
    def runs_path_parquet(self) -> Path:
        return self.results_dir / "runs.parquet"

    @property
    def runs_path_csv(self) -> Path:
        return self.results_dir / "runs.csv"

    @property
    def metrics_path_parquet(self) -> Path:
        return self.results_dir / "metrics.parquet"

    @property
    def metrics_path_csv(self) -> Path:
        return self.results_dir / "metrics.csv"

    # ---- per-run artifacts ----
    def meta_path(self, run_id: str) -> Path:
        return self.results_dir / "runs" / _clean(run_id) / "meta.json"

    def timeseries_paths(self, strategy: str, universe: str, run_id: str) -> tuple[Path, Path]:
        base = (
            self.results_dir
            / "timeseries"
            / f"strategy={_clean(strategy)}"
            / f"universe={_clean(universe)}"
            / f"run_id={_clean(run_id)}"
        )
        return base / "timeseries.parquet", base / "timeseries.csv"

    # ---- read helpers (single source of truth) ----
    def read_runs(self) -> pd.DataFrame:
        return _read_table(self.runs_path_csv, self.runs_path_parquet)

    def read_metrics(self) -> pd.DataFrame:
        return _read_table(self.metrics_path_csv, self.metrics_path_parquet)


def build_store(base_dir: Path | None = None, results_root: str = "results") -> ResultsStore:
    """
    Build a store rooted at repo root (or provided base_dir).
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[0]
    return ResultsStore(root=root, results_root=results_root)


# ----------------------------
# Public API for the dashboard
# ----------------------------

def load_runs(store: ResultsStore) -> pd.DataFrame:
    """
    Load runs and add a friendly label column.
    Keeps runs 'as-is' otherwise (no metrics join here).
    """
    runs = store.read_runs()
    if runs.empty:
        return runs

    runs = runs.copy()

    # Normalize expected columns
    for col in ["run_id", "strategy_name", "universe"]:
        if col not in runs.columns:
            runs[col] = ""

    runs["run_id"] = runs["run_id"].astype(str)
    runs["strategy_name"] = runs["strategy_name"].astype(str)
    runs["universe"] = runs["universe"].astype(str)

    runs["label"] = runs["run_id"] + " | " + runs["strategy_name"] + " | " + runs["universe"]

    if "created_at_utc" in runs.columns:
        runs = runs.sort_values("created_at_utc", ascending=False)

    return runs


def safe_read_metrics(store: ResultsStore) -> pd.DataFrame:
    try:
        return store.read_metrics()
    except Exception:
        return pd.DataFrame()


def load_runs_with_metrics(store: ResultsStore) -> pd.DataFrame:
    """
    Join runs + metrics (on run_id) so the table can sort/filter by Sharpe/CAGR/etc.
    """
    runs = load_runs(store)
    if runs.empty:
        return runs

    metrics = safe_read_metrics(store)
    if metrics.empty:
        return runs

    metrics = metrics.copy()
    if "run_id" not in metrics.columns:
        return runs

    metrics["run_id"] = metrics["run_id"].astype(str)

    # Avoid accidental column collisions; metrics wins only for metric-like names
    df = runs.merge(metrics, on="run_id", how="left", suffixes=("", "_metric"))
    return df


def read_meta_for_run(store: ResultsStore, run_id: str) -> dict[str, Any]:
    path = store.meta_path(run_id)
    if not path.exists():
        return {"info": f"meta.json not found for run_id={run_id}", "path": str(path)}

    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"info": "failed to read meta.json", "run_id": run_id, "error": str(e), "path": str(path)}


def meta_params_table(meta: Mapping[str, Any] | None) -> pd.DataFrame:
    if not isinstance(meta, dict) or not meta:
        return pd.DataFrame([{"parameter": "info", "value": "meta is empty"}])

    # prefer known param containers
    candidate: Any = None
    for key in ["params", "strategy_params", "model_params", "spec"]:
        if isinstance(meta.get(key), dict) and meta[key]:
            candidate = meta[key]
            break

    if candidate is None:
        candidate = {k: v for k, v in meta.items() if k not in ["run_id", "created_at", "created_at_utc"]}

    flat = _flatten(candidate)
    if not flat:
        return pd.DataFrame([{"parameter": "info", "value": "No parameters found in meta.json"}])

    rows = []
    for k, v in sorted(flat.items()):
        rows.append(
            {
                "parameter": k,
                "value": json.dumps(v) if isinstance(v, (list, dict)) else v,
            }
        )
    return pd.DataFrame(rows)


def read_timeseries_for_run(store: ResultsStore, run_id: str, time_col: str = "timestamp") -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns: (run_row: pd.Series, ts: pd.DataFrame)

    Steps:
      1) Load runs table and find row for run_id (strategy_name, universe)
      2) Read results/timeseries/strategy=<...>/universe=<...>/run_id=<...>/timeseries.(csv|parquet)
      3) Set time_col as DatetimeIndex
    """
    runs = load_runs(store)
    if runs.empty or "run_id" not in runs.columns:
        raise FileNotFoundError("runs table is missing or empty; cannot locate run metadata for timeseries.")

    match = runs.loc[runs["run_id"].astype(str) == str(run_id)]
    if match.empty:
        raise KeyError(f"run_id not found in runs table: {run_id}")

    row = match.iloc[0]
    strategy_name = str(row.get("strategy_name", ""))
    universe = str(row.get("universe", ""))

    pq, csv = store.timeseries_paths(strategy_name, universe, run_id)
    ts = _read_timeseries(csv, pq, time_col=time_col)

    if ts.empty:
        raise FileNotFoundError(
            f"Could not find timeseries for run_id={run_id}. Tried:\n"
            f"- {pq}\n"
            f"- {csv}"
        )

    return row, ts
