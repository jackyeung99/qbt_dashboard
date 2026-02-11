# registry.py
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Union

import yaml
from dash import Dash
from dash.development.base_component import Component


# ---- contracts --------------------------------------------------------------

class DashboardBuilder(Protocol):
    def __call__(self, ctx) -> Union["DashboardSpec", dict]: ...


@dataclass(frozen=True)
class DashboardSpec:
    """Fully-built dashboard plugin spec."""
    key: str
    title: str
    route: str
    description: str = ""
    tags: List[str] = None

    layout: Callable[[], Component] = None
    register_callbacks: Callable[[Dash], None] = None

    order: int = 100
    enabled: bool = True


@dataclass(frozen=True)
class DiscoveredDashboard:
    """Discovered dashboard with builder attached (ctx not applied yet)."""
    key: str
    title: str
    route: str
    description: str
    tags: List[str]
    order: int
    enabled: bool
    builder: DashboardBuilder


# ---- helpers ----------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _normalize_route(route: str) -> str:
    if not route:
        return "/"
    if not route.startswith("/"):
        route = "/" + route
    if route != "/" and route.endswith("/"):
        route = route[:-1]
    return route


def _default_boards_dir() -> Path:
    # registry.py is in repo root / dashboards root, boards/ is sibling
    return Path(__file__).resolve().parent / "boards"


def _resolve_package_base(package_base: Optional[str]) -> str:
    """
    Handles running as script vs module.

    If you run `python app.py` from this folder, importing `boards.*` works.
    If you run as a package (e.g. `python -m dashboards.app`), you likely want
    `dashboards.boards.*`.
    """
    if package_base:
        return package_base

    # Heuristic: if this file is part of a package import, __package__ will be non-empty.
    # If registry is imported as `dashboards.registry`, __package__ is "dashboards".
    pkg = __package__ or ""
    if pkg:
        return f"{pkg}.boards"
    return "boards"


# ---- public API -------------------------------------------------------------

def discover_dashboards(
    *,
    package_base: Optional[str] = None,
    boards_dir: Optional[Path] = None,
    enabled_keys: Optional[List[str]] = None,
) -> List[DiscoveredDashboard]:
    """
    Auto-discover dashboards under boards/*.

    Each dashboard folder must contain:
      - meta.yaml
      - dashboard.py (must define build_dashboard(ctx))

    enabled_keys:
      - None -> discover all
      - list -> only those keys
    """
    boards_dir = boards_dir or _default_boards_dir()
    pkg_base = _resolve_package_base(package_base)

    out: List[DiscoveredDashboard] = []

    for d in sorted([p for p in boards_dir.iterdir() if p.is_dir()]):
        key = d.name
        if enabled_keys is not None and key not in enabled_keys:
            continue

        meta_path = d / "meta.yaml"
        dash_py = d / "dashboard.py"
        if not meta_path.exists() or not dash_py.exists():
            continue

        meta = _load_yaml(meta_path)
        if meta.get("enabled", True) is False:
            continue

        route = _normalize_route(str(meta.get("route", f"/{key}")))
        title = str(meta.get("title", key))
        description = str(meta.get("description", ""))
        tags = list(meta.get("tags", []) or [])
        order = int(meta.get("order", 100))

        module_path = f"{pkg_base}.{key}.dashboard"
        mod = importlib.import_module(module_path)

        builder = getattr(mod, "build_dashboard", None)
        if builder is None or not callable(builder):
            raise ValueError(f"{module_path} must define a callable build_dashboard(ctx).")

        out.append(
            DiscoveredDashboard(
                key=key,
                title=title,
                route=route,
                description=description,
                tags=tags,
                order=order,
                enabled=True,
                builder=builder,
            )
        )

    out.sort(key=lambda s: (s.order, s.title.lower()))
    return out


def build_specs_with_ctx(discovered: List[DiscoveredDashboard], ctx) -> List[DashboardSpec]:
    """
    Materialize discovered dashboards into fully-built DashboardSpec by calling build_dashboard(ctx).
    """
    full: List[DashboardSpec] = []

    for d in discovered:
        built = d.builder(ctx)

        if isinstance(built, dict):
            layout_fn = built.get("layout")
            cb_fn = built.get("register_callbacks")
            title = built.get("title", d.title)
            route = _normalize_route(built.get("route", d.route))
        else:
            layout_fn = built.layout
            cb_fn = built.register_callbacks
            title = getattr(built, "title", d.title)
            route = _normalize_route(getattr(built, "route", d.route))

        if layout_fn is None:
            raise ValueError(f"Dashboard '{d.key}' build_dashboard(ctx) did not provide a layout().")
        if cb_fn is None:
            # allow dashboards with no callbacks
            cb_fn = lambda app: None

        full.append(
            DashboardSpec(
                key=d.key,
                title=title,
                route=route,
                description=d.description,
                tags=d.tags,
                order=d.order,
                enabled=d.enabled,
                layout=layout_fn,
                register_callbacks=cb_fn,
            )
        )

    return full


def specs_by_route(specs: List[DashboardSpec]) -> Dict[str, DashboardSpec]:
    out: Dict[str, DashboardSpec] = {}
    for s in specs:
        r = _normalize_route(s.route)
        if r in out:
            raise ValueError(f"Duplicate route detected: {r} from {s.key} and {out[r].key}")
        out[r] = s
    return out
