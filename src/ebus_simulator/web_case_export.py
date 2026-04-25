from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.models import PolyData, VolumeData
from ebus_simulator.rendering import build_render_context
from ebus_simulator.web_navigation import lps_to_web, preset_navigation_entries


SCHEMA_VERSION = 1
DEFAULT_MAX_MASK_POINTS = 1200
DEFAULT_MAX_STATION_POINTS = 900

ANATOMY_COLORS = {
    "airway": "#22c7c9",
    "lymph_node": "#93c56f",
    "station": "#92c774",
    "aorta": "#d13f3f",
    "atrial_appendage_left": "#ee7772",
    "left_atrial_appendage": "#ee7772",
    "azygous": "#2276c9",
    "brachiocephalic_trunk": "#d13f3f",
    "inferior_vena_cava": "#2276c9",
    "left_atrium": "#d13f3f",
    "left_brachiocephalic_vein": "#2276c9",
    "left_common_carotid_artery": "#d13f3f",
    "left_subclavian_artery": "#d13f3f",
    "left_ventricle": "#d13f3f",
    "pulmonary_artery": "#0b4f9f",
    "pulmonary_venous_system": "#ee7772",
    "right_atrium": "#2276c9",
    "right_brachiocephalic_vein": "#2276c9",
    "right_common_carotid_artery": "#d13f3f",
    "right_subclavian_artery": "#d13f3f",
    "right_ventricle": "#2276c9",
    "superior_vena_cava": "#2276c9",
    "esophagus": "#b79667",
}

DEFAULT_VESSEL_COLOR = "#2276c9"


@dataclass(frozen=True, slots=True)
class WebCaseExportResult:
    output_dir: str
    manifest_path: str
    preset_count: int
    airway_vertex_count: int
    airway_triangle_count: int
    vessel_asset_count: int
    station_asset_count: int
    clean_model_asset_count: int


class _Bounds:
    def __init__(self) -> None:
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def add_points_lps(self, points_lps: np.ndarray) -> None:
        if points_lps.size == 0:
            return
        points = np.asarray([lps_to_web(point) for point in np.asarray(points_lps, dtype=np.float64)], dtype=np.float64)
        self.add_points_web(points)

    def add_points_web(self, points_web: np.ndarray) -> None:
        if points_web.size == 0:
            return
        points = np.asarray(points_web, dtype=np.float64).reshape((-1, 3))
        current_min = np.min(points, axis=0)
        current_max = np.max(points, axis=0)
        if self._min is None or self._max is None:
            self._min = current_min
            self._max = current_max
            return
        self._min = np.minimum(self._min, current_min)
        self._max = np.maximum(self._max, current_max)

    def to_dict(self) -> dict[str, list[float]]:
        if self._min is None or self._max is None:
            minimum = np.zeros(3, dtype=np.float64)
            maximum = np.zeros(3, dtype=np.float64)
        else:
            minimum = self._min
            maximum = self._max
        center = (minimum + maximum) / 2.0
        size = maximum - minimum
        return {
            "min": [float(value) for value in minimum.tolist()],
            "max": [float(value) for value in maximum.tolist()],
            "center": [float(value) for value in center.tolist()],
            "size": [float(value) for value in size.tolist()],
        }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _web_points(points_lps: np.ndarray) -> list[list[float]]:
    return [lps_to_web(point) for point in np.asarray(points_lps, dtype=np.float64)]


def _triangulate_polygons(polygons: Iterable[np.ndarray]) -> list[list[int]]:
    triangles: list[list[int]] = []
    for polygon in polygons:
        indices = [int(value) for value in np.asarray(polygon, dtype=np.int64).tolist()]
        if len(indices) < 3:
            continue
        for offset in range(1, len(indices) - 1):
            triangles.append([indices[0], indices[offset], indices[offset + 1]])
    return triangles


def _mesh_payload(polydata: PolyData) -> dict[str, object]:
    triangles = _triangulate_polygons(polydata.polygons)
    return {
        "source_path": str(polydata.path),
        "source_space": polydata.source_space,
        "coordinate_frame": "web_xyz_mm_from_lps",
        "vertices": _web_points(polydata.points_lps),
        "triangles": triangles,
        "vertex_count": int(polydata.points_lps.shape[0]),
        "triangle_count": len(triangles),
    }


def _sample_mask_points_lps(mask_path: Path, *, max_points: int) -> tuple[np.ndarray, int]:
    volume = load_nifti(mask_path, kind="mask", load_data=True)
    return _sample_volume_points_lps(volume, max_points=max_points)


def _sample_volume_points_lps(volume: VolumeData, *, max_points: int) -> tuple[np.ndarray, int]:
    if volume.data is None:
        return np.empty((0, 3), dtype=np.float64), 0

    mask = np.asarray(volume.data) > 0
    flat = np.flatnonzero(mask)
    total = int(flat.size)
    if total == 0:
        return np.empty((0, 3), dtype=np.float64), 0

    sample_count = min(int(max_points), total)
    selected = flat if total <= sample_count else flat[np.linspace(0, total - 1, sample_count, dtype=np.int64)]
    ijk = np.column_stack(np.unravel_index(selected, volume.shape[:3])).astype(np.float64)
    homogeneous = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)], axis=1)
    points_lps = (volume.affine_lps @ homogeneous.T).T[:, :3]
    return points_lps.astype(np.float64), total


def _label_from_key(key: str) -> str:
    return key.replace("_", " ").title().replace("Svc", "SVC")


def _station_key(station: str) -> str:
    return f"station_{station.lower()}"


def _relative(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _clean_model_assets(clean_model_dir: str | Path | None, output_dir: Path) -> list[dict[str, object]]:
    if clean_model_dir is None:
        return []

    source_dir = Path(clean_model_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Clean model directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Clean model path is not a directory: {source_dir}")

    model_paths = sorted(source_dir.glob("*.glb"))
    if not model_paths:
        raise FileNotFoundError(f"No .glb clean model files found in: {source_dir}")

    preferred_name = "case_001.glb"
    primary_path = next((path for path in model_paths if path.name == preferred_name), model_paths[0])
    copied_assets: list[dict[str, object]] = []
    for model_path in model_paths:
        destination = output_dir / "models" / model_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, destination)
        copied_assets.append(
            {
                "key": model_path.stem,
                "label": model_path.stem.replace("_", " ").title(),
                "asset": _relative(destination, output_dir),
                "source_path": str(model_path),
                "coordinate_frame": "glb_scene_units_aligned_to_web_axes",
                "web_transform": "x=1000*x, y=1000*y, z=1000*z",
                "primary": model_path == primary_path,
            }
        )
    return copied_assets


def _scope_model_asset(scope_model_path: str | Path | None, output_dir: Path) -> dict[str, object] | None:
    if scope_model_path is None:
        return None

    source_path = Path(scope_model_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"EBUS scope model does not exist: {source_path}")
    if not source_path.is_file() or source_path.suffix.lower() != ".glb":
        raise ValueError(f"EBUS scope model must be a .glb file: {source_path}")

    destination = output_dir / "models" / "device" / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    is_tip_model = "tip" in source_path.stem.lower()
    if is_tip_model:
        label = "EBUS tip"
        shaft_axis = "+x"
        depth_axis = "+y"
        lateral_axis = "-z"
        fan_apex_anchor = {
            "x": "center",
            "y": "max",
            "z": "center",
        }
        fan_apex_anchor_point = [-0.334, -0.055, 0.0]
    else:
        label = "EBUS bronchoscope"
        shaft_axis = "+z"
        depth_axis = "+y"
        lateral_axis = "+x"
        fan_apex_anchor = {
            "x": "center",
            "y": "max",
            "z": "min",
        }
        fan_apex_anchor_point = None
    return {
        "key": source_path.stem,
        "label": label,
        "asset": _relative(destination, output_dir),
        "source_path": str(source_path),
        "coordinate_frame": "local_device_model_units",
        "shaft_axis": shaft_axis,
        "depth_axis": depth_axis,
        "lateral_axis": lateral_axis,
        "origin": "fan_apex_anchor_at_probe_contact",
        "fan_apex_anchor": fan_apex_anchor,
        "fan_apex_anchor_point": fan_apex_anchor_point,
        "scale_mm_per_unit": 44.0,
        "lock_to_fan": is_tip_model,
        "show_auxiliary_shaft": False,
    }


def attach_clean_model_assets(web_case_dir: str | Path, clean_model_dir: str | Path) -> int:
    output_root = Path(web_case_dir).expanduser().resolve()
    manifest_path = output_root / "case_manifest.web.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Web case manifest does not exist: {manifest_path}")

    payload = json.loads(manifest_path.read_text())
    assets = payload.setdefault("assets", {})
    if not isinstance(assets, dict):
        raise ValueError(f"Web case manifest has invalid assets payload: {manifest_path}")

    clean_assets = _clean_model_assets(clean_model_dir, output_root)
    assets["clean_models"] = clean_assets
    notes = payload.setdefault("notes", {})
    if isinstance(notes, dict):
        notes["clean_models"] = "GLB presentation meshes only; sector intersections still use source masks."
    _write_json(manifest_path, payload)
    return len(clean_assets)


def attach_scope_model_asset(web_case_dir: str | Path, scope_model_path: str | Path) -> bool:
    output_root = Path(web_case_dir).expanduser().resolve()
    manifest_path = output_root / "case_manifest.web.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Web case manifest does not exist: {manifest_path}")

    payload = json.loads(manifest_path.read_text())
    assets = payload.setdefault("assets", {})
    if not isinstance(assets, dict):
        raise ValueError(f"Web case manifest has invalid assets payload: {manifest_path}")

    assets["scope_model"] = _scope_model_asset(scope_model_path, output_root)
    notes = payload.setdefault("notes", {})
    if isinstance(notes, dict):
        notes["scope_model"] = "GLB bronchoscope tip presentation model aligned to the active probe pose."
    _write_json(manifest_path, payload)
    return assets["scope_model"] is not None


def export_web_case(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    max_mask_points: int = DEFAULT_MAX_MASK_POINTS,
    max_station_points: int = DEFAULT_MAX_STATION_POINTS,
    clean_model_dir: str | Path | None = None,
    scope_model_path: str | Path | None = None,
) -> WebCaseExportResult:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    context = build_render_context(manifest_path)
    bounds = _Bounds()

    airway_mesh = context.airway_display_mesh or context.airway_geometry_mesh
    if airway_mesh is None:
        raise ValueError("The manifest does not provide an airway mesh for web export.")

    airway_asset_path = output_root / "geometry" / "airway_mesh.json"
    airway_payload = _mesh_payload(airway_mesh)
    _write_json(airway_asset_path, airway_payload)
    bounds.add_points_lps(airway_mesh.points_lps)

    centerline_asset_path = output_root / "geometry" / "centerlines.json"
    centerline_payload = {
        "source_path": context.main_graph.source_path,
        "coordinate_frame": "web_xyz_mm_from_lps",
        "polylines": [
            {
                "line_index": int(polyline.line_index),
                "points": _web_points(polyline.points_lps),
                "cumulative_lengths_mm": [float(value) for value in polyline.cumulative_lengths_mm.tolist()],
                "total_length_mm": float(polyline.total_length_mm),
                "point_count": int(polyline.points_lps.shape[0]),
            }
            for polyline in sorted(context.main_graph.polylines, key=lambda item: item.line_index)
        ],
    }
    if not centerline_payload["polylines"]:
        raise ValueError("The main centerline graph has no exportable polylines.")
    longest = max(centerline_payload["polylines"], key=lambda item: float(item["total_length_mm"]))
    centerline_payload["primary_line_index"] = int(longest["line_index"])
    centerline_payload["primary_total_length_mm"] = float(longest["total_length_mm"])
    _write_json(centerline_asset_path, centerline_payload)
    for polyline in context.main_graph.polylines:
        bounds.add_points_lps(polyline.points_lps)

    navigation_entries = preset_navigation_entries(context)
    used_vessel_keys = sorted({key for entry in navigation_entries for key in entry.vessel_overlays if key in context.manifest.overlay_masks})
    used_station_keys = sorted({_station_key(entry.station) for entry in navigation_entries})

    vessel_assets: list[dict[str, object]] = []
    for vessel_key in used_vessel_keys:
        points_lps, source_count = _sample_mask_points_lps(
            context.manifest.overlay_masks[vessel_key],
            max_points=max_mask_points,
        )
        asset_path = output_root / "geometry" / "vessels" / f"{vessel_key}_points.json"
        payload = {
            "key": vessel_key,
            "label": _label_from_key(vessel_key),
            "coordinate_frame": "web_xyz_mm_from_lps",
            "source_path": str(context.manifest.overlay_masks[vessel_key]),
            "source_voxel_count": int(source_count),
            "point_count": int(points_lps.shape[0]),
            "points": _web_points(points_lps),
        }
        _write_json(asset_path, payload)
        bounds.add_points_lps(points_lps)
        vessel_assets.append(
            {
                "key": vessel_key,
                "label": _label_from_key(vessel_key),
                "asset": _relative(asset_path, output_root),
                "color": ANATOMY_COLORS.get(vessel_key, DEFAULT_VESSEL_COLOR),
                "source_voxel_count": int(source_count),
                "point_count": int(points_lps.shape[0]),
            }
        )

    station_assets: list[dict[str, object]] = []
    for station_key in used_station_keys:
        if station_key not in context.manifest.station_masks:
            continue
        points_lps, source_count = _sample_mask_points_lps(
            context.manifest.station_masks[station_key],
            max_points=max_station_points,
        )
        asset_path = output_root / "geometry" / "stations" / f"{station_key}_points.json"
        payload = {
            "key": station_key,
            "label": f"{_label_from_key(station_key)} lymph node region",
            "coordinate_frame": "web_xyz_mm_from_lps",
            "source_path": str(context.manifest.station_masks[station_key]),
            "source_voxel_count": int(source_count),
            "point_count": int(points_lps.shape[0]),
            "points": _web_points(points_lps),
        }
        _write_json(asset_path, payload)
        bounds.add_points_lps(points_lps)
        station_assets.append(
            {
                "key": station_key,
                "label": f"{_label_from_key(station_key)} lymph node region",
                "asset": _relative(asset_path, output_root),
                "color": ANATOMY_COLORS["station"],
                "source_voxel_count": int(source_count),
                "point_count": int(points_lps.shape[0]),
            }
        )

    preset_payloads: list[dict[str, object]] = []
    node_assets: list[dict[str, object]] = []
    for entry in navigation_entries:
        contact_lps = np.asarray(entry.contact_lps, dtype=np.float64)
        target_lps = np.asarray(entry.target_lps, dtype=np.float64)
        bounds.add_points_lps(np.vstack([contact_lps, target_lps]))
        station_asset = next((asset["asset"] for asset in station_assets if asset["key"] == _station_key(entry.station)), None)
        preset_payloads.append(
            {
                **asdict(entry),
                "label": f"Station {entry.station.upper()} node {entry.node.upper()} ({entry.approach})",
                "station_key": _station_key(entry.station),
                "contact": lps_to_web(contact_lps),
                "target": lps_to_web(target_lps),
                "shaft_axis": None if entry.shaft_axis_lps is None else lps_to_web(entry.shaft_axis_lps),
                "depth_axis": None if entry.depth_axis_lps is None else lps_to_web(entry.depth_axis_lps),
                "lateral_axis": None if entry.lateral_axis_lps is None else lps_to_web(entry.lateral_axis_lps),
                "station_asset": station_asset,
            }
        )
        node_assets.append(
            {
                "key": f"{entry.preset_key}:lymph_node",
                "preset_key": entry.preset_key,
                "station_key": _station_key(entry.station),
                "label": f"Station {entry.station.upper()} lymph node",
                "position": lps_to_web(target_lps),
                "position_lps": entry.target_lps,
                "radius_mm": max(4.5, min(9.5, entry.contact_to_target_distance_mm * 0.18)),
                "color": ANATOMY_COLORS["lymph_node"],
            }
        )

    render_defaults = context.manifest.render_defaults
    clean_model_assets = _clean_model_assets(clean_model_dir, output_root)
    scope_model_asset = _scope_model_asset(scope_model_path, output_root)
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": context.manifest.case_id,
        "source_manifest": str(Path(manifest_path).expanduser().resolve()),
        "coordinate_frame": {
            "source": "LPS_mm",
            "web": "x=L, y=S, z=-P",
        },
        "render_defaults": {
            "sector_angle_deg": float(render_defaults.get("sector_angle_deg", 60.0)),
            "max_depth_mm": float(render_defaults.get("max_depth_mm", 40.0)),
            "roll_deg": float(render_defaults.get("roll_deg", 0.0)),
        },
        "bounds": bounds.to_dict(),
        "assets": {
            "airway_mesh": _relative(airway_asset_path, output_root),
            "centerlines": _relative(centerline_asset_path, output_root),
            "vessels": vessel_assets,
            "stations": station_assets,
            "clean_models": clean_model_assets,
            "scope_model": scope_model_asset,
        },
        "navigation": {
            "mode": "guided_centerline",
            "primary_line_index": int(centerline_payload["primary_line_index"]),
            "primary_total_length_mm": float(centerline_payload["primary_total_length_mm"]),
        },
        "presets": preset_payloads,
        "anatomy": {
            "nodes": node_assets,
        },
        "color_map": {
            **ANATOMY_COLORS,
            **{asset["key"]: asset["color"] for asset in vessel_assets},
        },
        "notes": {
            "intent": "local anatomy-correlation teaching app",
            "mask_assets": "translucent point-cloud fallbacks for v1 browser performance",
            "clean_models": "GLB presentation meshes only; sector intersections still use source masks.",
            "scope_model": "GLB bronchoscope tip presentation model aligned to the active probe pose.",
            "navigation": "guided along exported centerline polylines with curated station snaps",
        },
    }

    manifest_asset_path = output_root / "case_manifest.web.json"
    _write_json(manifest_asset_path, manifest_payload)

    return WebCaseExportResult(
        output_dir=str(output_root),
        manifest_path=str(manifest_asset_path),
        preset_count=len(preset_payloads),
        airway_vertex_count=int(airway_payload["vertex_count"]),
        airway_triangle_count=int(airway_payload["triangle_count"]),
        vessel_asset_count=len(vessel_assets),
        station_asset_count=len(station_assets),
        clean_model_asset_count=len(clean_model_assets),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a browser-friendly EBUS anatomy correlation case.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--output-dir", required=True, help="Directory for web case JSON and geometry assets.")
    parser.add_argument("--max-mask-points", type=int, default=DEFAULT_MAX_MASK_POINTS, help="Maximum sampled points per vessel mask.")
    parser.add_argument("--max-station-points", type=int, default=DEFAULT_MAX_STATION_POINTS, help="Maximum sampled points per station mask.")
    parser.add_argument("--clean-model-dir", help="Optional directory of clean GLB presentation models to copy into the web case.")
    parser.add_argument("--scope-model", help="Optional EBUS bronchoscope tip GLB to copy into the web case.")
    args = parser.parse_args()

    result = export_web_case(
        args.manifest,
        output_dir=args.output_dir,
        max_mask_points=args.max_mask_points,
        max_station_points=args.max_station_points,
        clean_model_dir=args.clean_model_dir,
        scope_model_path=args.scope_model,
    )
    print(f"web_case: {result.output_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"presets: {result.preset_count}")
    print(f"airway_vertices: {result.airway_vertex_count}")
    print(f"airway_triangles: {result.airway_triangle_count}")
    print(f"vessel_assets: {result.vessel_asset_count}")
    print(f"station_assets: {result.station_asset_count}")
    print(f"clean_model_assets: {result.clean_model_asset_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
