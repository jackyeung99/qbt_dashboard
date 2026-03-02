# app.py  (repo root / dashboards package root)
from __future__ import annotations

import os

from dash import Dash, html

# ---- import that works both as module and as script ----
try:
    # if you run: python -m dashboards.app (package import)
    from .registry import discover_dashboards, build_specs_with_ctx  # type: ignore
except Exception:
    # if you run: python app.py (script from this folder)
    from registry import discover_dashboards, build_specs_with_ctx  # type: ignore


class AppContext:
    def __init__(self, env: str = "dev", results_root: str = "results"):
        self.env = env
        self.results_root = results_root



def create_app(ctx: AppContext, dashboard) -> Dash:
    key = dashboard

    discovered = discover_dashboards(enabled_keys=[key])
    specs = build_specs_with_ctx(discovered, ctx)

    if not specs:
        # show available dashboards if missing
        available = [p.key for p in build_specs_with_ctx(discover_dashboards(enabled_keys=None), ctx)]
        raise ValueError(
            f"Dashboard '{key}' not found under boards/{key}.\n"
            f"Available dashboards: {available}\n"
            f"Set env: DASHBOARD=<one of the above>"
        )

    if len(specs) > 1:
        # should not happen if enabled_keys=[key], but guard anyway
        raise ValueError(f"Expected exactly 1 dashboard for key='{key}', got {len(specs)}: {[s.key for s in specs]}")

    spec = specs[0]

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title=(spec.title or "Dashboard"),
    )

    # register callbacks once
    if spec.register_callbacks:
        spec.register_callbacks(app)

    # DO NOT force padding if your dashboard layout already does its own styling.
    # Keep the wrapper minimal.
    app.layout =  spec.layout()

    return app


ctx = AppContext(
    env=os.getenv("APP_ENV", "dev"),
    results_root=os.getenv("RESULTS_ROOT", "results")
)
app = create_app(ctx, dashboard='macro')
server = app.server

if __name__ == "__main__":
    app.run(debug="1")
