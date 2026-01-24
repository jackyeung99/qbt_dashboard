from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


# ----------------------------
# Simple filesystem-based "store"
# ----------------------------

@dataclass(frozen=True)
class ResultsStore:
    """Read-only access to exported dashboard artifacts under <root>/results."""
    root: Path  # points to repo root (or any base); results are under root / "results"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

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

    def meta_path(self, run_id: str) -> Path:
        return self.results_dir / "runs" / run_id / "meta.json"

    # --- optional: timeseries paths ---
    def timeseries_path_flat(self, run_id: str) -> tuple[Path, Path]:
        """Convention A: results/timeseries/<run_id>.(parquet|csv)"""
        base = self.results_dir / "timeseries"
        return base / f"{run_id}.parquet", base / f"{run_id}.csv"

    def timeseries_path_partitioned(self, strategy_name: str, universe: str, run_id: str) -> tuple[Path, Path]:
        """Convention B: results/timeseries/strategy=<...>/universe=<...>/<run_id>.(parquet|csv)"""
        base = self.results_dir / "timeseries" / f"strategy={strategy_name}" / f"universe={universe}"
        return base / f"{run_id}.parquet", base / f"{run_id}.csv"


def _read_table(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """
    Prefer CSV if present (deployment-friendly). Fall back to parquet if present.
    Return empty df if neither exists.
    """
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        # If you removed parquet deps for deployment, ensure you shipped CSV instead.
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


# ----------------------------
# Public API (mirrors your old functions)
# ----------------------------

def build_store(base_dir: Path | None = None) -> ResultsStore:
    """
    Build a store rooted at the repo root. Default: directory containing this file.
    If your dashboard module lives elsewhere, pass base_dir=Path(".") from app.py.
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[0]
    return ResultsStore(root=root)


def load_runs(store: ResultsStore) -> pd.DataFrame:
    runs = _read_table(store.runs_path_csv, store.runs_path_parquet)
    if runs.empty:
        return runs

    runs = runs.copy()

    # Normalize expected columns (be forgiving)
    for col in ["run_id", "strategy_name", "universe"]:
        if col not in runs.columns:
            runs[col] = ""

    # created_at_utc sorting if available
    sort_col = "created_at_utc" if "created_at_utc" in runs.columns else None

    runs["label"] = runs["run_id"].astype(str) + " | " + runs["strategy_name"].astype(str) + " | " + runs["universe"].astype(str)

    if sort_col:
        return runs.sort_values(sort_col, ascending=False)
    return runs


def safe_read_metrics(store: ResultsStore) -> pd.DataFrame:
    try:
        return _read_table(store.metrics_path_csv, store.metrics_path_parquet)
    except Exception:
        return pd.DataFrame()


def read_meta_for_run(store: ResultsStore, run_id: str) -> dict:
    path = store.meta_path(run_id)
    if not path.exists():
        return {"info": f"meta.json not found for run_id={run_id}", "path": str(path)}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"info": "failed to read meta.json", "run_id": run_id, "error": str(e), "path": str(path)}


def _flatten(d, prefix=""):
    """Flatten nested dict into dot keys: {'a': {'b': 1}} -> {'a.b': 1}"""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def meta_params_table(meta: dict) -> pd.DataFrame:
    if not isinstance(meta, dict) or not meta:
        return pd.DataFrame([{"parameter": "info", "value": "meta is empty"}])

    # Try known param containers first
    for key in ["params", "strategy_params", "model_params", "spec"]:
        if isinstance(meta.get(key), dict) and meta[key]:
            candidate = meta[key]
            break
    else:
        # fallback: use entire meta minus obvious non-params
        candidate = {k: v for k, v in meta.items() if k not in ["run_id", "created_at", "created_at_utc"]}

    flat = _flatten(candidate)
    if not flat:
        return pd.DataFrame([{"parameter": "info", "value": "No parameters found in meta.json"}])

    return pd.DataFrame(
        [{"parameter": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
         for k, v in sorted(flat.items())]
    )


def read_timeseries_for_run(store: ResultsStore, run_id: str):
    """
    Returns: (run_row: pd.Series, ts: pd.DataFrame)

    Timeseries lookup strategy:
    1) Load runs table, find the row (strategy_name, universe).
    2) Try flat convention: results/timeseries/<run_id>.(csv|parquet)
    3) Try partitioned convention: results/timeseries/strategy=<...>/universe=<...>/<run_id>.(csv|parquet)
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

    # Try convention A (flat)
    pq, csv = store.timeseries_path_flat(run_id)
    ts = _read_table(csv, pq)
    if not ts.empty:
        return row, ts

    # Try convention B (partitioned)
    pq, csv = store.timeseries_path_partitioned(strategy_name, universe, run_id)
    ts = _read_table(csv, pq)
    if not ts.empty:
        return row, ts

    raise FileNotFoundError(
        f"Could not find timeseries for run_id={run_id}. Tried:\n"
        f"- {store.timeseries_path_flat(run_id)[0]} / {store.timeseries_path_flat(run_id)[1]}\n"
        f"- {store.timeseries_path_partitioned(strategy_name, universe, run_id)[0]} / {store.timeseries_path_partitioned(strategy_name, universe, run_id)[1]}"
    )
