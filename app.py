from dash import Dash

from config import CONFIG
from layout.main import make_layout
from services.data_access import build_store
from callbacks.main import register_callbacks


def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[CONFIG.theme])
    app.layout = make_layout(CONFIG.max_width_px)

    store_ctx = build_store(CONFIG.base_dir)
    register_callbacks(app, store_ctx)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
