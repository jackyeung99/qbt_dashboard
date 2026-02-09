from dash import Dash

from functools import lru_cache
from pathlib import Path
from config import CONFIG
from layout.main import make_layout
from services.data_access import build_store
from callbacks.main import register_callbacks
from experiment import make_app

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results" / "xle_rv_sweep"



@lru_cache(maxsize=1)
def load_data():
    runs_summary = pd.read_parquet(RESULTS / "runs_summary.parquet")
    # equity_curves = pd.read_parquet(RESULTS / "equity_curves.parquet")
    thresholds = pd.read_parquet(RESULTS / "thresholds.parquet")
    # returns = pd.read_parquet(RESULTS / "returns.parquet")[['run_id', 'timestamp', 'split', 'rvol_o2c', 'ret_cc', 'ret_oc']]
    return runs_summary, thresholds

def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[CONFIG.theme])
    # app.layout = make_layout(CONFIG.max_width_px)

    # store_ctx = build_store(CONFIG.base_dir)
    # register_callbacks(app, store_ctx)

    runs_summary, thresholds = load_data()

    app = make_app(runs_summary, thresholds)
    return app


app = create_app()
server = app.server  

if __name__ == "__main__":
    app.run(debug=True)




