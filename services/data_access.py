from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


def _clean(x: str) -> str:
    """
    Keep filesystem-safe names similar to your original _clean().
    """
    return str(x).strip().replace("/", "_").replace("\\", "_")


# ----------------------------
# Simple filesystem-based "store" (mirrors StoragePaths layout)
# ----------------------------

@dataclass(frozen=True)
class ResultsStore:
    """
    Read-only access to exported dashboard artifacts under <root>/results.

    Mirrors your original StoragePaths layout:
      results/
        runs.(csv|parquet)
        metrics.(csv|parquet)
        runs/<run_id>/meta.json
        timeseries/strategy=<strategy>/universe=<universe>/run_id=<run_id>/timeseries.(csv|parquet)
    """
    root: Path

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

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

    def timeseries_path(self, strategy: str, universe: str, run_id: str) -> tuple[Path, Path]:
        """
        Matches original StoragePaths.run_timeseries_key(...)/timeseries.parquet, but as filesystem Paths.

        results/timeseries/strategy=<strategy>/universe=<universe>/run_id=<run_id>/timeseries.(parquet|csv)
        """
        base = (
            self.results_dir
            / "timeseries"
            / f"strategy={_clean(strategy)}"
            / f"universe={_clean(universe)}"
            / f"run_id={_clean(run_id)}"
        )
        return base / "timeseries.parquet", base / "timeseries.csv"


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

def _read_timeseries(csv_path: Path, parquet_path: Path, time_col: str = "date") -> pd.DataFrame:
    """
    Read timeseries and set timestamp column as DatetimeIndex.
    Assumes exported CSV has a column like 'date' or 'timestamp'.
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=[time_col])
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        return pd.DataFrame()

    # If time column exists, set as index
    if time_col in df.columns:
        df = df.set_index(time_col)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    return df


# ----------------------------
# Public API (mirrors your old functions)
# ----------------------------

def build_store(base_dir: Path | None = None) -> ResultsStore:
    """
    Build a store rooted at the repo root.

    Recommended usage:
        store = build_store(Path("."))

    If base_dir is None, default to directory containing this module.
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

    runs["label"] = (
        runs["run_id"].astype(str)
        + " | "
        + runs["strategy_name"].astype(str)
        + " | "
        + runs["universe"].astype(str)
    )

    if "created_at_utc" in runs.columns:
        runs = runs.sort_values("created_at_utc", ascending=False)

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
        return {
            "info": "failed to read meta.json",
            "run_id": run_id,
            "error": str(e),
            "path": str(path),
        }


def _flatten(d, prefix: str = "") -> dict:
    """Flatten nested dict into dot keys: {'a': {'b': 1}} -> {'a.b': 1}"""
    out: dict = {}
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
    candidate = None
    for key in ["params", "strategy_params", "model_params", "spec"]:
        if isinstance(meta.get(key), dict) and meta[key]:
            candidate = meta[key]
            break

    if candidate is None:
        # fallback: use entire meta minus obvious non-params
        candidate = {k: v for k, v in meta.items() if k not in ["run_id", "created_at", "created_at_utc"]}

    flat = _flatten(candidate)
    if not flat:
        return pd.DataFrame([{"parameter": "info", "value": "No parameters found in meta.json"}])

    return pd.DataFrame(
        [
            {
                "parameter": k,
                "value": json.dumps(v) if isinstance(v, (list, dict)) else v,
            }
            for k, v in sorted(flat.items())
        ]
    )


def read_timeseries_for_run(store: ResultsStore, run_id: str):
    """
    Returns: (run_row: pd.Series, ts: pd.DataFrame)

    Lookup:
      1) Load runs table, find the row (strategy_name, universe)
      2) Read results/timeseries/strategy=<...>/universe=<...>/run_id=<...>/timeseries.(csv|parquet)
      3) Set timestamp as DatetimeIndex
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

    pq, csv = store.timeseries_path(strategy_name, universe, run_id)

    ts = _read_timeseries(csv, pq, time_col="timestamp")
    if ts.empty:
        raise FileNotFoundError(
            f"Could not find timeseries for run_id={run_id}. Tried:\n"
            f"- {pq}\n"
            f"- {csv}"
        )

    return row, ts