from __future__ import annotations

from typing import Any, Dict, Tuple
from dash import Dash
from dash.development.base_component import Component

from .queries import load_data
from .layout import build_layout  # you’ll update build_layout signature
from .callbacks import register_callbacks
from datetime import datetime
import pytz  # or zoneinfo in Python 3.9+


TZ = pytz.timezone("America/Indiana/Indianapolis") 

def _daily_key():
    # changes once per day (in your timezone)
    return datetime.now(TZ).strftime("%Y-%m-%d")


def build_dashboard(ctx) -> Dict[str, Any]:
    """
    Registry entrypoint.
    Returns a dict with:
      - title
      - route (optional)
      - layout() -> Dash Component
      - register_callbacks(app) -> None
    """

    # -----------------------------------------
    # Data loader (kept as callable for reuse)
    # -----------------------------------------
    def load_live_data() -> Tuple:
        # returns (equity_df, performance_dict, meta_dict)
        return load_data(_daily_key())

    def layout() -> Component:
      return build_layout(title="Live Portfolio", subtitle="XLE StateSignal")

    def _register(app: Dash) -> None:
        register_callbacks(app, load_live_data=load_live_data)

    return {
        "title": "Live Portfolio",
        "route": "",  # optional
        "layout": layout,
        "register_callbacks": _register,
    }
