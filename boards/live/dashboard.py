from __future__ import annotations

from typing import Any, Dict, Tuple
from dash import Dash
from dash.development.base_component import Component

from .queries import load_data, extract_etfs_from_weights, pick_default_etf
from .layout import build_layout  # you’ll update build_layout signature
from .callbacks import register_callbacks
from datetime import datetime
import pytz  # or zoneinfo in Python 3.9+

TZ = pytz.timezone("America/Indiana/Indianapolis") 


ETF_LABEL_MAP = {
    "XLE": "Energy (XLE)",
    "XLC": "Communication Services (XLC)",
    "XLY": "Consumer Discretionary (XLY)",
    "XLP": "Consumer Staples (XLP)",
    "XLF": "Financials (XLF)",
    "XLV": "Health Care (XLV)",
    "XLI": "Industrials (XLI)",
    "XLB": "Materials (XLB)",
    "XLRE": "Real Estate (XLRE)",
    "XLK": "Technology (XLK)",
    "XLU": "Utilities (XLU)",
}

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

    def load_live_data() -> Tuple:
        return load_data(_daily_key())

    def layout() -> Component:
        equity_df, meta = load_live_data()

        etfs = extract_etfs_from_weights(equity_df) if equity_df is not None and not equity_df.empty else []

        etf_options = [
            {
                "label": ETF_LABEL_MAP.get(etf, etf),  # fallback if missing
                "value": etf,
            }
            for etf in etfs
        ]

        default_etf = pick_default_etf(etfs)

        return build_layout(
            title="Live Portfolio",
            etf_options=etf_options,
            default_etf=default_etf,
        )

    def _register(app: Dash) -> None:
        register_callbacks(app, load_live_data=load_live_data)

    return {
        "title": "Live Portfolio",
        "route": "",
        "layout": layout,
        "register_callbacks": _register,
    }