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
    strategy_options = [
        {"label": "Sector ETF Long Only", "value": "sector_long_only"},
        {"label": "XLE Long Only", "value": "long_only"},
        {"label": "XLE Long / Short", "value": "long_short"},
    ]

    default_strategy = "sector_long_only"

    def load_live_data(strategy_key: str | None = None) -> Tuple:
        strategy_key = strategy_key or default_strategy

        return load_data(
            cache_key=f"{_daily_key()}-{strategy_key}",
            strategy_key=strategy_key,
        )

    def get_etf_options(strategy_key: str | None = None):
        equity_df, _meta = load_live_data(strategy_key)

        etfs = (
            extract_etfs_from_weights(equity_df)
            if equity_df is not None and not equity_df.empty
            else []
        )

        etf_options = [
            {
                "label": ETF_LABEL_MAP.get(etf, etf),
                "value": etf,
            }
            for etf in etfs
        ]

        default_etf = pick_default_etf(etfs)

        return etf_options, default_etf

    def layout() -> Component:
        etf_options, default_etf = get_etf_options(default_strategy)

        return build_layout(
            title="Live Portfolio",
            strategy_options=strategy_options,
            default_strategy=default_strategy,
            etf_options=etf_options,
            default_etf=default_etf,
        )

    def _register(app: Dash) -> None:
        register_callbacks(
            app,
            load_live_data=load_live_data,
            get_etf_options=get_etf_options,
        )

    return {
        "title": "Live Portfolio",
        "route": "",
        "layout": layout,
        "register_callbacks": _register,
    }