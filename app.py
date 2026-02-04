from dash import Dash

from config import CONFIG
from layout.main import make_layout
from services.data_access import build_store
from callbacks.main import register_callbacks
from experiment import make_app

import pandas as pd

def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[CONFIG.theme])
    # app.layout = make_layout(CONFIG.max_width_px)

    # store_ctx = build_store(CONFIG.base_dir)
    # register_callbacks(app, store_ctx)

    runs_summary = pd.read_parquet('results/xle_rv_sweep/runs_summary.parquet')
    equity_curves = pd.read_parquet('results/xle_rv_sweep/equity_curves.parquet')
    thresholds = pd.read_parquet('results/xle_rv_sweep/thresholds.parquet')

    app = make_app(runs_summary, equity_curves, thresholds)
    return app


app = create_app()
server = app.server  

if __name__ == "__main__":
    app.run(debug=True)
