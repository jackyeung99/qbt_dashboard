# src/dashboards/registry.py
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml
from dash import Dash
from dash.development.base_component import Component


@dataclass()
class DashboardSpec:
    """Single dashboard plugin spec."""
    key: str                 # folder name, e.g. "energy_overview"
    title: str               # display name
    route: str               # URL path, e.g. "/energy"
    description: str = ""
    tags: List[str] = None

    # factories
    layout: Callable[[], Component] = None
    register_callbacks: Callable[[Dash], None] = None

    # optional settings
    order: int = 100
    enabled: bool = True


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _normalize_route(route: str) -> str:
    if not route:
        return "/"
    if not route.startswith("/"):
        route = "/" + route
    # no trailing slash except root
    if route != "/" and route.endswith("/"):
        route = route[:-1]
    return route


def discover_dashboards(
    *,
    package_base: str = "boards",
    boards_dir: Optional[Path] = None,
    enabled_keys: Optional[List[str]] = None,
) -> List[DashboardSpec]:
    """
    Auto-discover dashboards under boards/*.

    Each dashboard folder must contain:
      - meta.yaml
      - dashboard.py  (must expose build_dashboard(ctx) -> DashboardSpec-like fields)

    enabled_keys:
      - None -> load all
      - list -> only load those dashboard folder names
    """
    if boards_dir is None:
        boards_dir = Path(__file__).resolve().parent / "boards"

    specs: List[DashboardSpec] = []

    for d in sorted([p for p in boards_dir.iterdir() if p.is_dir()]):
        
        # print(d)
       
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

        module_path = f"{package_base}.{key}.dashboard"
        mod = importlib.import_module(module_path)


        if not hasattr(mod, "build_dashboard"):
            raise ValueError(f"{module_path} must define build_dashboard(ctx).")
      

        # build_dashboard should return an object/dict with layout + register_callbacks.
        # We keep it flexible: allow dict or DashboardSpec.
        built = mod.build_dashboard  # function handle; called later with ctx


        # We store a "lazy" spec; ctx is not known at discovery time.
        specs.append(
            DashboardSpec(
                key=key,
                title=title,
                route=route,
                description=description,
                tags=tags,
                order=order,
                enabled=True,
                # placeholders; app.py will replace these by calling build_dashboard(ctx)
                layout=lambda: None,  # type: ignore
                register_callbacks=lambda app: None,  # type: ignore
            )
        )

        # Attach the builder on the spec (private attribute) for later use
        setattr(specs[-1], "_builder", built)  # type: ignore[attr-defined]

    # sort by order then title
    specs.sort(key=lambda s: (s.order, s.title.lower()))
    return specs


def build_specs_with_ctx(specs: List[DashboardSpec], ctx) -> List[DashboardSpec]:
    """
    Convert 'lazy' discovered specs into full specs by calling each dashboard's build_dashboard(ctx).
    """
    full: List[DashboardSpec] = []
    for s in specs:
        builder = getattr(s, "_builder", None)
        if builder is None:
            continue

        built = builder(ctx)

        # allow dict return
        if isinstance(built, dict):
            layout_fn = built["layout"]
            cb_fn = built["register_callbacks"]
        else:
            layout_fn = built.layout
            cb_fn = built.register_callbacks

        full.append(
            DashboardSpec(
                key=s.key,
                title=getattr(built, "title", s.title) if not isinstance(built, dict) else built.get("title", s.title),
                route=getattr(built, "route", s.route) if not isinstance(built, dict) else built.get("route", s.route),
                description=s.description,
                tags=s.tags,
                order=s.order,
                enabled=s.enabled,
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
