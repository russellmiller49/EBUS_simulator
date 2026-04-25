from __future__ import annotations

import argparse
import json
from pathlib import Path
import webbrowser

from ebus_simulator.centerline import CenterlinePolyline
from ebus_simulator.web_case_export import attach_clean_model_assets, attach_scope_model_asset, export_web_case
from ebus_simulator.web_navigation import (
    build_navigation_response,
    preset_navigation_entries,
)
from ebus_simulator.web_volume_intersections import (
    DEFAULT_DEPTH_SAMPLES,
    DEFAULT_LATERAL_SAMPLES,
    DEFAULT_SLAB_HALF_THICKNESS_MM,
    DEFAULT_SLAB_SAMPLES,
    build_volume_sector_response,
)
from ebus_simulator.rendering import build_render_context


DEFAULT_WEB_CASE_DIR = Path("reports/web_case")
DEFAULT_SCOPE_MODEL_PATHS = (Path("model/EBUS_tip.glb"), Path("model/EBUS_bronchoscope.glb"))
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _load_fastapi():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "launch-web-app requires the optional web dependencies. "
            "Install them with `python -m pip install -e '.[web]'`."
        ) from exc
    return FastAPI, HTTPException, FileResponse, HTMLResponse, JSONResponse, StaticFiles


def _safe_asset_path(root: Path, asset_path: str) -> Path:
    candidate = (root / asset_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"Asset path escapes the web case directory: {asset_path!r}")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(asset_path)
    return candidate


def _load_case_manifest(web_case_dir: Path) -> dict[str, object]:
    manifest_path = web_case_dir / "case_manifest.web.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Web case manifest does not exist: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _centerlines_from_context(context) -> dict[int, CenterlinePolyline]:
    return {int(polyline.line_index): polyline for polyline in context.main_graph.polylines}


def _default_scope_model_path() -> Path | None:
    for path in DEFAULT_SCOPE_MODEL_PATHS:
        if path.exists():
            return path.resolve()
    return None


def _default_index_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EBUS Anatomy Correlation</title>
  <style>
    body { margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101415; color: #eef4f2; }
    main { min-height: 100vh; display: grid; place-items: center; padding: 32px; box-sizing: border-box; }
    section { max-width: 720px; border: 1px solid rgba(255,255,255,.16); background: rgba(255,255,255,.05); padding: 24px; border-radius: 8px; }
    code { color: #94d1ff; }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>EBUS Anatomy Correlation Backend Is Running</h1>
      <p>The FastAPI case server is ready. Build the web frontend with <code>cd web && npm install && npm run build</code>, then restart <code>launch-web-app</code> to serve the app from this same URL.</p>
      <p>API check: <a href="/api/case">/api/case</a></p>
    </section>
  </main>
</body>
</html>
"""


def create_app(manifest_path: str | Path, *, web_case_dir: str | Path):
    FastAPI, HTTPException, FileResponse, HTMLResponse, JSONResponse, StaticFiles = _load_fastapi()

    resolved_web_case = Path(web_case_dir).expanduser().resolve()
    case_manifest = _load_case_manifest(resolved_web_case)
    context = build_render_context(manifest_path)
    centerlines = _centerlines_from_context(context)
    presets = preset_navigation_entries(context)
    presets_by_key = {preset.preset_key: preset for preset in presets}

    defaults = case_manifest.get("render_defaults", {})
    default_depth = float(defaults.get("max_depth_mm", 40.0)) if isinstance(defaults, dict) else 40.0
    default_sector = float(defaults.get("sector_angle_deg", 60.0)) if isinstance(defaults, dict) else 60.0
    assets = case_manifest.get("assets", {})
    station_keys = [
        str(asset.get("key"))
        for asset in (assets.get("stations", []) if isinstance(assets, dict) else [])
        if isinstance(asset, dict) and asset.get("key")
    ]
    vessel_keys = [
        str(asset.get("key"))
        for asset in (assets.get("vessels", []) if isinstance(assets, dict) else [])
        if isinstance(asset, dict) and asset.get("key")
    ]
    color_map = case_manifest.get("color_map", {})
    resolved_color_map = color_map if isinstance(color_map, dict) else {}

    app = FastAPI(title="EBUS Anatomy Correlation Simulator", version="0.1.0")

    @app.get("/api/case")
    def api_case():
        return JSONResponse(case_manifest)

    @app.get("/api/asset/{asset_path:path}")
    def api_asset(asset_path: str):
        try:
            path = _safe_asset_path(resolved_web_case, asset_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(path)

    @app.get("/api/navigation")
    def api_navigation(
        preset_key: str | None = None,
        line_index: int | None = None,
        s_mm: float | None = None,
        roll_deg: float = 0.0,
        max_depth_mm: float | None = None,
        sector_angle_deg: float | None = None,
    ):
        preset = presets_by_key.get(preset_key) if preset_key else (presets[0] if presets else None)
        resolved_line_index = int(line_index if line_index is not None else (preset.line_index if preset is not None else sorted(centerlines)[0]))
        resolved_s_mm = float(s_mm if s_mm is not None else (preset.centerline_s_mm if preset is not None else 0.0))
        try:
            return JSONResponse(
                build_navigation_response(
                    centerlines_by_index=centerlines,
                    preset=preset,
                    line_index=resolved_line_index,
                    centerline_s_mm=resolved_s_mm,
                    roll_deg=float(roll_deg),
                    max_depth_mm=float(default_depth if max_depth_mm is None else max_depth_mm),
                    sector_angle_deg=float(default_sector if sector_angle_deg is None else sector_angle_deg),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sector-volume")
    def api_sector_volume(
        preset_key: str | None = None,
        line_index: int | None = None,
        s_mm: float | None = None,
        roll_deg: float = 0.0,
        max_depth_mm: float | None = None,
        sector_angle_deg: float | None = None,
        slab_half_thickness_mm: float = DEFAULT_SLAB_HALF_THICKNESS_MM,
        depth_samples: int = DEFAULT_DEPTH_SAMPLES,
        lateral_samples: int = DEFAULT_LATERAL_SAMPLES,
        slab_samples: int = DEFAULT_SLAB_SAMPLES,
    ):
        preset = presets_by_key.get(preset_key) if preset_key else (presets[0] if presets else None)
        resolved_line_index = int(line_index if line_index is not None else (preset.line_index if preset is not None else sorted(centerlines)[0]))
        resolved_s_mm = float(s_mm if s_mm is not None else (preset.centerline_s_mm if preset is not None else 0.0))
        try:
            return JSONResponse(
                build_volume_sector_response(
                    context,
                    centerlines_by_index=centerlines,
                    preset=preset,
                    line_index=resolved_line_index,
                    centerline_s_mm=resolved_s_mm,
                    roll_deg=float(roll_deg),
                    max_depth_mm=float(default_depth if max_depth_mm is None else max_depth_mm),
                    sector_angle_deg=float(default_sector if sector_angle_deg is None else sector_angle_deg),
                    station_keys=station_keys,
                    vessel_keys=vessel_keys,
                    color_map=resolved_color_map,
                    depth_samples=int(depth_samples),
                    lateral_samples=int(lateral_samples),
                    slab_half_thickness_mm=float(slab_half_thickness_mm),
                    slab_samples=int(slab_samples),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    dist_dir = Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="web")
    else:
        @app.get("/")
        def index():
            return HTMLResponse(_default_index_html())

    return app


def launch_web_app(
    manifest_path: str | Path,
    *,
    web_case_dir: str | Path = DEFAULT_WEB_CASE_DIR,
    clean_model_dir: str | Path | None = None,
    scope_model_path: str | Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "launch-web-app requires uvicorn. Install optional dependencies with "
            "`python -m pip install -e '.[web]'`."
        ) from exc

    resolved_web_case = Path(web_case_dir).expanduser().resolve()
    resolved_scope_model = Path(scope_model_path).expanduser().resolve() if scope_model_path is not None else None
    if resolved_scope_model is None:
        resolved_scope_model = _default_scope_model_path()

    if not (resolved_web_case / "case_manifest.web.json").exists():
        export_web_case(
            manifest_path,
            output_dir=resolved_web_case,
            clean_model_dir=clean_model_dir,
            scope_model_path=resolved_scope_model,
        )
    elif clean_model_dir is not None:
        attach_clean_model_assets(resolved_web_case, clean_model_dir)
    if resolved_scope_model is not None:
        attach_scope_model_asset(resolved_web_case, resolved_scope_model)

    url = f"http://{host}:{int(port)}"
    print(f"Serving EBUS anatomy correlation app at {url}")
    print(f"Web case: {resolved_web_case}")
    if open_browser:
        webbrowser.open(url)

    uvicorn.run(
        create_app(manifest_path, web_case_dir=resolved_web_case),
        host=host,
        port=int(port),
        reload=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the local browser EBUS anatomy correlation simulator.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--web-case", default=str(DEFAULT_WEB_CASE_DIR), help="Exported web case directory. Auto-exported if missing.")
    parser.add_argument("--clean-model-dir", help="Optional directory of clean GLB presentation models to attach to the web case.")
    parser.add_argument("--scope-model", help="Optional EBUS tip GLB to attach to the web case. Defaults to model/EBUS_tip.glb, then model/EBUS_bronchoscope.glb, if present.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host interface. Default: {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port. Default: {DEFAULT_PORT}.")
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser after launch.")
    args = parser.parse_args()

    launch_web_app(
        args.manifest,
        web_case_dir=args.web_case,
        clean_model_dir=args.clean_model_dir,
        scope_model_path=args.scope_model,
        host=args.host,
        port=args.port,
        open_browser=args.open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
