from __future__ import annotations

from dash import dcc, html


THEME = {
    "colors": {
        "page_bg": "#f6f7fb",
        "surface": "#ffffff",
        "surface_alt": "#fbfcfe",
        "border": "#e6e6e6",
        "border_soft": "#d6d9e0",
        "text": "#111827",
        "text_muted": "#6b7280",
        "text_soft": "#4b5563",
        "heading": "#1f2937",
        "badge_live_bg": "#d1e7dd",
        "badge_live_text": "#0f5132",
        "badge_neutral_bg": "#f2f3f7",
        "shadow": "0 1px 2px rgba(0,0,0,0.04)",
    },
    "font": {
        "family": "system-ui",
        "title": "32px",
        "subtitle": "16px",
        "section": "16px",
        "kpi_label": "12px",
        "kpi_value": "20px",
        "badge": "18px",
        "body": "18px",
    },
    "space": {
        "page_pad": "16px",
        "container_pad": "16px",
        "section_gap": "12px",
        "card_pad": "12px",
        "panel_gap": "10px",
        "title_gap": "4px",
        "kpi_gap": "12px",
        "kpi_label_gap": "6px",
        "header_y": "14px 0",
        "placeholder_pad": "16px",
    },
    "radius": {
        "card": "14px",
        "inner": "12px",
        "pill": "999px",
    },
    "layout": {
        "max_width": "1900px",
        "kpi_row_cols": "repeat(6, minmax(0, 1fr))",
        "positioning_cols": "repeat(2, minmax(0, 1fr))",
        "top_grid_cols": "1fr",
        "overview_grid_cols": "2fr 1fr",
        "etf_kpi_cols": "repeat(4, minmax(0, 1fr))",
        "etf_detail_cols": "2fr 1fr",
        "overview_min_height": "420px",
        "detail_min_height": "360px",
        "dropdown_min_width": "240px",
    },
}


STRATEGY_OVERVIEW_MD = """
This strategy is based on the idea that asset returns behave differently across volatility regimes.  
We use volatility as a state variable to classify trading days into favorable and unfavorable environments, and adjust how much we invest accordingly.

The goal is to identify a threshold that separates periods where the asset earns higher risk-adjusted returns from periods where it does not, allowing us to condition investment decisions on the observed volatility regime.
"""

REALIZED_VOL_MD = r"""
We use intraday realized volatility on day $t$ as a proxy for the asset's latent daily volatility. We can compute this by first computing the log returns from 5-min
intraday returns between market open (9:30 AM ET) to when we make the decision and place trades (3:45 PM ET). 

$$
r_{t,i} = \log\left(\frac{P_{t,i}}{P_{t,i-1}}\right)
$$

We then take the sum of squares to get Realized Variance 
$$
RV_t = \sum_i r_{t,i}^2
$$

as well as realized volatility
$$
RVOL_t = \sqrt{RV_t}.
$$
"""

REGIME_CLASSIFICATION_MD = r"""
Based on the realized volatility observed on day $t$, we classify each day into a volatility regime using a threshold $\tau^*$.

The underlying idea is that market behavior differs between stable and turbulent environments, and this can be captured by our volatility proxy.  
By identifying the prevailing regime, we can condition trading decisions and dynamically adjust portfolio exposure based on current risk conditions.

- **Low Volatility Regime**
  - Markets are more stable  
  - Trends tend to persist  
  - Lower risk 

- **High Volatility Regime**
  - Markets are more uncertain  
  - Large drawdowns are more likely  
  - Higher risk

Formally, the regime is defined as:

$$
\text{Regime}_t =
\begin{cases}
\text{Low Volatility}, & RVOL_t < \tau^* \\
\text{High Volatility}, & RVOL_t \ge \tau^*
\end{cases}
$$

Where: 
- $\tau^*$ is an empirically determined threshold that separates low- and high-volatility environments  

"""

THRESHOLD_SELECTION_MD = r"""
The threshold \( \tau^* \) is estimated on a rolling 2-year training window using a linear scan over a grid of 100 candidate values.

### Grid Construction

We first construct a set of candidate thresholds  $\{\tau_j\}_{j=1}^{100}$ over the empirical distribution of realized volatility.

To ensure robustness, the realized volatility series is **winsorized at the 5% level**, so that extreme values are clipped.  
The grid is then defined over the central range:

$$
\tau_j \in [\text{P}_{5}(RVOL), \ \text{P}_{95}(RVOL)]
$$

This ensures that the candidate thresholds focus on typical market conditions rather than rare outliers specifically it 
- Reduces sensitivity to extreme volatility spikes
- Prevents unstable threshold estimates across rolling windows
- Focuses the model on persistent, tradable regimes



### Objective Function

For each candidate threshold $\tau_j$, we split the data into two regimes:
- Low-volatility: $RVOL_t \leq \tau_j$
- High-volatility: $RVOL_t > \tau_j$

Within each regime, we compute Sharpe ratios and define:

$$
\Delta(\tau_j) = SR_{\text{low}}(\tau_j) - SR_{\text{high}}(\tau_j)
$$


### Threshold Selection

We select the threshold that maximizes the separation:

$$
\tau^* = \arg\max_{\tau_j} \Delta(\tau_j)
$$

This identifies the volatility cutoff that best distinguishes favorable and unfavorable environments.




"""

INVESTMENT_RULE_MD = r"""
After estimating \( \tau^* \), we evaluate the current day’s realized volatility and define a trading signal:

$$
s_t = \mathbf{1}\{RVOL_t \leq \tau^*\}
$$

We define two ways of allocating resources for a given strategy 

### Binary Allocation

In the simplest case, the signal directly determines exposure:

$$
w_t = w_{\max} \cdot s_t
$$

where:
- $w_{\max}$ is the maximum allowable allocation to the asset

This implies:
- $s_t = 1$ → fully invested ($w_t = w_{\max}$)  
- $s_t = 0$ → no investment ($w_t = 0$)  


### Mean-Variance Allocation

Alternatively, the signal can be used to partially invest based on the regime and the 2 year historical window.

We consider a one-dimensional weight $w \in [w_{\text{low}}, w_{\text{high}}]$, representing the allocation to the asset (with the remainder held in cash).

We then perform a simple linear scan over candidate weights:

$$
w \in \{w_1, w_2, \dots, w_K\}, \quad w_k \in [w_{\text{low}}, w_{\text{high}}]
$$

For each candidate weight $w_k$, we evaluate a mean-variance objective using regime-specific estimates of expected return $\mu$ and variance $\sigma^2$:

$$
\max_{w_k} \left( w_k \mu - \frac{\gamma}{2} w_k^2 \sigma^2 \right)
$$

The optimal weight is then selected as:

$$
w_t = \arg\max_{w_k \in [w_{\text{low}}, w_{\text{high}}]} \left( w_k \mu - \frac{\gamma}{2} w_k^2 \sigma^2 \right)
$$

This procedure is performed separately for each regime:
- if $s_t = 1$, we use parameters estimated from low-volatility periods  
- if $s_t = 0$, we use parameters estimated from high-volatility periods  

"""

MULTI_ASSET_MD = r"""
The framework extends naturally to multiple assets by allocating capital across independent strategies.

Given $m$ assets, we partition the total capital so that each strategy operates on an equal share:

$$
\text{Capital per strategy} = \frac{1}{m}
$$

Each asset-specific strategy is then run independently using its own:
- realized volatility $RVOL_{i,t}$
- threshold $\tau_i^*$
- signal $s_{i,t}$

and produces its own allocation:
$$
w_{i,t} = \frac{1}{m} \cdot \tilde{w}_{i,t}
$$

where $\tilde{w}_{i,t}$ is the weight determined by the single-asset strategy (e.g., binary or mean-variance).
"""


CHOSEN_ASSETS_MD = r"""

We consider ETFs as our primary investment instruments, as they provide natural diversification across sectors.

The strategy was initially developed and tested on the Energy Select Sector ETF ($XLE$).  
We then extend the framework to a broader universe of sector ETFs to evaluate its performance across different parts of the market.

Specifically, we run the strategy on the following sector ETFs, which together cover the major sectors of the S\&P 500:

Each ETF represents a distinct sector:
- Energy ($XLE$)
- Communication Services ($XLC$)
- Consumer Discretionary ($XLY$)
- Consumer Staples ($XLP$)
- Financials ($XLF$)
- Health Care ($XLV$)
- Industrials ($XLI$)
- Materials ($XLB$)
- Technology ($XLK$)
- Utilities ($XLU$)

This setup allows us to test whether the volatility-regime framework generalizes across sectors with different risk profiles and economic sensitivities.


"""

STRATEGY_CONTENT = [
    ("Strategy Overview", STRATEGY_OVERVIEW_MD),
    ("Realized Volatility", REALIZED_VOL_MD),
    ("Regime Classification", REGIME_CLASSIFICATION_MD),
    ("Threshold Selection τ*", THRESHOLD_SELECTION_MD),
    ("Investment Rule", INVESTMENT_RULE_MD),
    ("Multiple Assets", MULTI_ASSET_MD),
    ("Selected Assets", CHOSEN_ASSETS_MD),
]


def sx_card() -> dict:
    c = THEME["colors"]
    r = THEME["radius"]
    return {
        "background": c["surface"],
        "border": f'1px solid {c["border"]}',
        "borderRadius": r["card"],
        "boxShadow": c["shadow"],
        "boxSizing": "border-box",
    }


def sx_panel() -> dict:
    s = THEME["space"]
    return {
        **sx_card(),
        "padding": s["card_pad"],
        "display": "flex",
        "flexDirection": "column",
        "gap": s["panel_gap"],
        "overflow": "hidden",
    }


def sx_section_title() -> dict:
    c = THEME["colors"]
    f = THEME["font"]
    return {
        "fontWeight": 800,
        "fontSize": f["section"],
        "marginBottom": THEME["space"]["panel_gap"],
        "color": c["heading"],
    }


def sx_placeholder() -> dict:
    c = THEME["colors"]
    r = THEME["radius"]
    s = THEME["space"]
    return {
        "width": "100%",
        "flex": "1",
        "minHeight": 0,
        "border": f'1px dashed {c["border_soft"]}',
        "borderRadius": r["inner"],
        "background": c["surface_alt"],
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "textAlign": "center",
        "color": c["text_muted"],
        "padding": s["placeholder_pad"],
        "boxSizing": "border-box",
        "overflow": "hidden",
    }


def page_shell(children) -> html.Div:
    return html.Div(
        style={
            "fontFamily": THEME["font"]["family"],
            "background": THEME["colors"]["page_bg"],
            "minHeight": "100vh",
            "width": "100%",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                style={
                    "maxWidth": THEME["layout"]["max_width"],
                    "width": "100%",
                    "margin": "0 auto",
                    "padding": THEME["space"]["container_pad"],
                    "boxSizing": "border-box",
                },
                children=children,
            )
        ],
    )


def kpi_card(title: str, value_id: str) -> html.Div:
    c = THEME["colors"]
    f = THEME["font"]
    s = THEME["space"]

    return html.Div(
        style={
            **sx_card(),
            "padding": s["card_pad"],
            "minWidth": 0,
            "minHeight": "74px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
        },
        children=[
            html.Div(
                title,
                style={
                    "fontSize": f["kpi_label"],
                    "color": c["text_muted"],
                    "marginBottom": s["kpi_label_gap"],
                },
            ),
            html.Div(
                id=value_id,
                style={
                    "fontSize": f["kpi_value"],
                    "fontWeight": 800,
                    "color": c["text"],
                },
            ),
        ],
    )


def info_badge(
    text: str | None = None,
    *,
    id: str | None = None,
    live: bool = False,
) -> html.Div:
    c = THEME["colors"]
    f = THEME["font"]

    bg = c["badge_live_bg"] if live else c["badge_neutral_bg"]
    fg = c["badge_live_text"] if live else c["text_soft"]
    border = "none" if live else f'1px solid {c["border"]}'

    return html.Div(
        text,
        id=id,
        style={
            "fontSize": f["badge"],
            "color": fg,
            "background": bg,
            "padding": "6px 10px",
            "borderRadius": THEME["radius"]["pill"],
            "fontWeight": 700 if live else 600,
            "whiteSpace": "nowrap",
            "border": border,
        },
    )


def section_card(title: str, children, *, style: dict | None = None) -> html.Div:
    return html.Div(
        style={**sx_card(), "padding": THEME["space"]["card_pad"], **(style or {})},
        children=[
            html.Div(title, style=sx_section_title()),
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": THEME["space"]["section_gap"],  
                },
                children=children,
            ),
        ],
    )


def graph_panel(graph_id: str, *, min_height: str, figure: dict | None = None) -> html.Div:
    return html.Div(
        style={**sx_panel(), "minHeight": min_height},
        children=[
            dcc.Graph(
                id=graph_id,
                figure=figure or {},
                config={"displayModeBar": False},
                style={"width": "100%", "height": "100%"},
            ),
        ],
    )


def content_panel(panel_id: str, *, min_height: str, padding: str = "14px") -> html.Div:
    return html.Div(
        id=panel_id,
        style={**sx_panel(), "minHeight": min_height, "padding": padding},
    )


def collapsible_section(title: str, content: str, *, open: bool = False) -> html.Details:
    c = THEME["colors"]
    f = THEME["font"]

    return html.Details(
        open=open,
        style={
            "border": f'1px solid {c["border"]}',
            "borderRadius": THEME["radius"]["inner"],
            "padding": "10px 14px",
            "background": c["surface_alt"],
            "marginBottom": "10px",
        },
        children=[
            html.Summary(
                title,
                style={
                    "cursor": "pointer",
                    "fontWeight": 700,
                    "fontSize": f["section"],
                    "color": c["text"],
                },
            ),
            dcc.Markdown(
                content,
                mathjax=True,
                style={
                    "fontSize": f["body"],
                    "lineHeight": 1.6,
                    "color": c["text_soft"],
                    "marginTop": "10px",
                },
            ),
        ],
    )


def strategy_card() -> html.Div:
    return section_card(
        "Strategy Methodology",
        [
            html.Div(
                style={"display": "flex", "flexDirection": "column", "gap": "0px"},
                children=[
                    collapsible_section(title, content, open=(i == 0))
                    for i, (title, content) in enumerate(STRATEGY_CONTENT)
                ],
            )
        ],
        style={"marginBottom": THEME["space"]["section_gap"]},
    )


def build_header(*, title: str, subtitle: str | None) -> html.Div:
    c = THEME["colors"]
    f = THEME["font"]
    s = THEME["space"]

    return html.Div(
        style={
            "padding": s["header_y"],
            "marginBottom": s["section_gap"],
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "flex-start",
            "gap": s["section_gap"],
        },
        children=[
            html.Div(
                children=[
                    html.Div(
                        title,
                        style={
                            "margin": "0",
                            "fontSize": f["title"],
                            "fontWeight": 800,
                            "letterSpacing": "-0.2px",
                            "color": c["text"],
                        },
                    ),
                    html.Div(
                        subtitle or "Live monitoring for a multi-ETF volatility regime portfolio.",
                        style={
                            "marginTop": s["title_gap"],
                            "fontSize": f["subtitle"],
                            "color": c["text_muted"],
                        },
                    ),
                ]
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                },
                children=[
                    info_badge("Live", id="live_status_badge", live=True),
                    info_badge(id="generated_at"),
                ],
            ),
        ],
    )


def build_portfolio_section() -> html.Div:
    s = THEME["space"]
    l = THEME["layout"]

    portfolio_kpis = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": l["kpi_row_cols"],
            "gap": s["kpi_gap"],
        },
        children=[
            kpi_card("Number of Trading Days", "kpi-n-obs"),
            kpi_card("Total PNL", "kpi-total-pnl"),
            kpi_card("Sharpe", "kpi-sharpe"),
            kpi_card("CAGR", "kpi-cagr"),
            kpi_card("Max Drawdown", "kpi-mdd"),
            kpi_card("Avg. Daily Return", "kpi-mean-return"),
        ],
    )

    portfolio_top_row = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": l["overview_grid_cols"],
            "gap": s["section_gap"],
        },
        children=[
            graph_panel("equity-fig", min_height=l["overview_min_height"]),
            content_panel("portfolio-stats-panel", min_height=l["detail_min_height"]),
        ],
    )

    portfolio_bottom_row = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": s["section_gap"],
        },
        children=[
            graph_panel("allocation-fig", min_height=l["overview_min_height"]),
            graph_panel("weights-stats-panel", min_height=l["overview_min_height"]),
        ],
    )

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": l["top_grid_cols"],
            "gap": s["section_gap"],
            "marginBottom": s["section_gap"],
        },
        children=[
            section_card(
                "Portfolio Performance",
                [
                    portfolio_kpis,
                    portfolio_top_row,
                    portfolio_bottom_row,
                ],
            )
        ],
    )


def build_asset_section(
    *,
    etf_options: list[dict] | None = None,
    default_etf: str | None = None,
) -> html.Div:
    s = THEME["space"]
    l = THEME["layout"]

    asset_controls = html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": s["section_gap"],
            "gap": s["section_gap"],
            "flexWrap": "wrap",
        },
        children=[
            html.Div("", style={"display": "none"}),
            dcc.Dropdown(
                id="etf-selector",
                options=etf_options or [],
                value=default_etf,
                placeholder="Select ETF",
                clearable=False,
                style={"minWidth": l["dropdown_min_width"]},
            ),
        ],
    )

    asset_kpis = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": l["etf_kpi_cols"],
            "gap": s["kpi_gap"],
            "marginBottom": s["section_gap"],
        },
        children=[
            kpi_card("ETF Return", "etf-kpi-return"),
            kpi_card("ETF Sharpe", "etf-kpi-sharpe"),
            kpi_card("Current Regime", "etf-kpi-regime"),
            kpi_card("Target Weight", "etf-kpi-weight"),
        ],
    )

    asset_detail_row = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": l["etf_detail_cols"],
            "gap": s["section_gap"],
        },
        children=[
            graph_panel("etf-main-fig", min_height=l["detail_min_height"]),
            content_panel("etf-side-panel", min_height=l["detail_min_height"]),
        ],
    )

    return section_card(
        "Asset Information",
        [
            asset_controls,
            asset_kpis,
            asset_detail_row,
        ],
        style={"marginBottom": s["section_gap"]},
    )


def build_layout(
    *,
    title: str = "Live Portfolio",
    subtitle: str | None = None,
    etf_options: list[dict] | None = None,
    default_etf: str | None = None,
):
    s = THEME["space"]

    return page_shell(
        [
            dcc.Location(id="url"),
            build_header(title=title, subtitle=subtitle),
            html.Div(
                style={
                    "display": "grid",
                    "gap": s["section_gap"],
                    "marginBottom": s["section_gap"],
                },
                children=[strategy_card()],
            ),
            build_portfolio_section(),
            build_asset_section(
                etf_options=etf_options,
                default_etf=default_etf,
            ),
        ]
    )