from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from ebus_simulator.centerline import CenterlinePolyline
from ebus_simulator.manifest import resolve_preset_overrides
from ebus_simulator.models import ManifestPreset


EPSILON = 1e-9
DEFAULT_TANGENT_WINDOW_MM = 5.0
DEFAULT_SECTOR_SLAB_HALF_THICKNESS_MM = 4.0
SUPERIOR_AXIS_LPS = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class WebNavigationPose:
    line_index: int
    centerline_s_mm: float
    position_lps: list[float]
    tangent_lps: list[float]
    depth_axis_lps: list[float]
    lateral_axis_lps: list[float]
    roll_deg: float


@dataclass(frozen=True, slots=True)
class WebPresetNavigation:
    preset_key: str
    preset_id: str
    station: str
    node: str
    approach: str
    line_index: int
    centerline_s_mm: float
    contact_lps: list[float]
    target_lps: list[float]
    shaft_axis_lps: list[float] | None
    depth_axis_lps: list[float] | None
    lateral_axis_lps: list[float] | None
    vessel_overlays: list[str]
    contact_to_target_distance_mm: float


def lps_to_web(point_lps: np.ndarray | list[float] | tuple[float, float, float]) -> list[float]:
    point = np.asarray(point_lps, dtype=np.float64)
    return [float(point[0]), float(point[2]), float(-point[1])]


def web_to_lps(point_web: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    point = np.asarray(point_web, dtype=np.float64)
    return np.asarray([point[0], -point[2], point[1]], dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return None
    return vector / norm


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return vector - (np.dot(vector, axis) * axis)


def _orthonormal_pose_axes(shaft_axis: np.ndarray, depth_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    shaft = _normalize(shaft_axis)
    if shaft is None:
        return None
    depth = _normalize(_project_perpendicular(depth_axis, shaft))
    if depth is None:
        return None
    lateral = _normalize(np.cross(shaft, depth))
    if lateral is None:
        return None
    depth = _normalize(np.cross(lateral, shaft))
    if depth is None:
        return None
    return depth, lateral


def _choose_fallback_depth_axis(shaft_axis: np.ndarray) -> np.ndarray:
    candidates = [
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    ]
    ordered = sorted(candidates, key=lambda item: abs(float(np.dot(item, shaft_axis))))
    for candidate in ordered:
        normalized = _normalize(_project_perpendicular(candidate, shaft_axis))
        if normalized is not None:
            return normalized
    raise ValueError("Failed to construct a fallback depth axis.")


def _rotate_around_axis(vector: np.ndarray, axis: np.ndarray, roll_deg: float) -> np.ndarray:
    if abs(roll_deg) <= EPSILON:
        return vector.copy()

    axis_unit = _normalize(axis)
    if axis_unit is None:
        raise ValueError("Rotation axis is undefined.")

    angle = math.radians(float(roll_deg))
    return (
        vector * math.cos(angle)
        + np.cross(axis_unit, vector) * math.sin(angle)
        + axis_unit * np.dot(axis_unit, vector) * (1.0 - math.cos(angle))
    )


def _tangent_at_s(polyline: CenterlinePolyline, centerline_s_mm: float, *, window_mm: float = DEFAULT_TANGENT_WINDOW_MM) -> np.ndarray:
    if polyline.total_length_mm <= EPSILON:
        raise ValueError(f"Centerline polyline {polyline.line_index} has no usable length.")

    clamped = float(np.clip(centerline_s_mm, 0.0, polyline.total_length_mm))
    start_s = max(0.0, clamped - window_mm)
    end_s = min(polyline.total_length_mm, clamped + window_mm)
    if end_s - start_s <= EPSILON:
        start_s = max(0.0, clamped - (window_mm * 2.0))
        end_s = min(polyline.total_length_mm, clamped + (window_mm * 2.0))

    tangent = polyline.point_at_arc_length(end_s) - polyline.point_at_arc_length(start_s)
    normalized = _normalize(tangent)
    if normalized is not None:
        return normalized

    if polyline.points_lps.shape[0] >= 2:
        fallback = _normalize(polyline.points_lps[-1] - polyline.points_lps[0])
        if fallback is not None:
            return fallback
    raise ValueError(f"Centerline polyline {polyline.line_index} tangent is undefined.")


def navigation_pose_from_polyline(
    polyline: CenterlinePolyline,
    *,
    centerline_s_mm: float,
    roll_deg: float = 0.0,
    target_lps: np.ndarray | None = None,
    contact_lps: np.ndarray | None = None,
    contact_centerline_s_mm: float | None = None,
    shaft_axis_lps: np.ndarray | None = None,
    depth_axis_lps: np.ndarray | None = None,
) -> WebNavigationPose:
    centerline_position = polyline.point_at_arc_length(centerline_s_mm).astype(np.float64)
    position = centerline_position.copy()
    tangent = _tangent_at_s(polyline, centerline_s_mm)
    preset_shaft = None if shaft_axis_lps is None else _normalize(np.asarray(shaft_axis_lps, dtype=np.float64))
    if preset_shaft is not None:
        if float(np.dot(tangent, preset_shaft)) < 0.0:
            tangent = -tangent
        if contact_centerline_s_mm is not None and abs(float(centerline_s_mm) - float(contact_centerline_s_mm)) <= 1.0:
            tangent = preset_shaft

    if contact_lps is not None and contact_centerline_s_mm is not None:
        contact = np.asarray(contact_lps, dtype=np.float64)
        reference_centerline_position = polyline.point_at_arc_length(contact_centerline_s_mm).astype(np.float64)
        radial_offset = _project_perpendicular(contact - reference_centerline_position, tangent)
        radial_length = float(np.linalg.norm(radial_offset))
        if radial_length > EPSILON:
            if abs(float(centerline_s_mm) - float(contact_centerline_s_mm)) <= 1.0:
                position = contact
            else:
                position = centerline_position + (radial_offset / radial_length) * radial_length

    raw_depth: np.ndarray | None = None
    if depth_axis_lps is not None:
        raw_depth = _normalize(_project_perpendicular(np.asarray(depth_axis_lps, dtype=np.float64), tangent))
    if raw_depth is None and target_lps is not None:
        raw_depth = _normalize(_project_perpendicular(np.asarray(target_lps, dtype=np.float64) - position, tangent))
    if raw_depth is None:
        raw_depth = _choose_fallback_depth_axis(tangent)
    if target_lps is not None and float(np.dot(np.asarray(target_lps, dtype=np.float64) - position, raw_depth)) < 0.0:
        raw_depth = -raw_depth

    depth_axis = _normalize(_rotate_around_axis(raw_depth, tangent, roll_deg))
    if depth_axis is None:
        raise ValueError("Depth axis is undefined.")

    axes = _orthonormal_pose_axes(tangent, depth_axis)
    if axes is None:
        raise ValueError("Pose axes could not be re-orthogonalized.")
    depth_axis, lateral_axis = axes

    return WebNavigationPose(
        line_index=int(polyline.line_index),
        centerline_s_mm=float(np.clip(centerline_s_mm, 0.0, polyline.total_length_mm)),
        position_lps=[float(value) for value in position.tolist()],
        tangent_lps=[float(value) for value in tangent.tolist()],
        depth_axis_lps=[float(value) for value in depth_axis.tolist()],
        lateral_axis_lps=[float(value) for value in lateral_axis.tolist()],
        roll_deg=float(roll_deg),
    )


def _preset_by_id(presets: list[ManifestPreset]) -> dict[str, ManifestPreset]:
    return {preset.id: preset for preset in presets}


def preset_key(preset_id: str, approach: str) -> str:
    return f"{preset_id}::{approach}"


def preset_navigation_entries(context) -> list[WebPresetNavigation]:
    presets = _preset_by_id(context.manifest.presets)
    entries: list[WebPresetNavigation] = []

    for pose in sorted(context.pose_report.poses, key=lambda item: (item.preset_id, item.contact_approach)):
        preset = presets[pose.preset_id]
        query = pose.centerline_query
        if query is None:
            projection = context.main_graph.nearest_point(np.asarray(pose.contact_world, dtype=np.float64))
            line_index = 0 if projection is None else int(projection.line_index)
            centerline_s_mm = 0.0 if projection is None else float(projection.line_arclength_mm)
        else:
            line_index = int(query.line_index) if query.line_index is not None else 0
            centerline_s_mm = 0.0 if query.line_arclength_mm is None else float(query.line_arclength_mm)

        overrides = resolve_preset_overrides(preset, approach=pose.contact_approach)
        vessel_overlays = [] if overrides is None or overrides.vessel_overlays is None else list(overrides.vessel_overlays)

        entries.append(
            WebPresetNavigation(
                preset_key=preset_key(pose.preset_id, pose.contact_approach),
                preset_id=str(pose.preset_id),
                station=str(pose.station),
                node=str(pose.node),
                approach=str(pose.contact_approach),
                line_index=line_index,
                centerline_s_mm=centerline_s_mm,
                contact_lps=[float(value) for value in pose.contact_world],
                target_lps=[float(value) for value in pose.target_world],
                shaft_axis_lps=(None if pose.shaft_axis is None else [float(value) for value in pose.shaft_axis]),
                depth_axis_lps=(None if pose.depth_axis is None else [float(value) for value in pose.depth_axis]),
                lateral_axis_lps=(None if pose.lateral_axis is None else [float(value) for value in pose.lateral_axis]),
                vessel_overlays=vessel_overlays,
                contact_to_target_distance_mm=float(pose.contact_to_target_distance_mm),
            )
        )

    return entries


def cephalic_image_axis_lps(pose: WebNavigationPose) -> np.ndarray:
    tangent = np.asarray(pose.tangent_lps, dtype=np.float64)
    normalized = _normalize(tangent)
    if normalized is None:
        raise ValueError("Pose tangent is undefined.")
    return normalized if float(np.dot(normalized, SUPERIOR_AXIS_LPS)) >= 0.0 else -normalized


def sector_plane_normal_lps(pose: WebNavigationPose) -> np.ndarray:
    image_axis = cephalic_image_axis_lps(pose)
    depth_axis = np.asarray(pose.depth_axis_lps, dtype=np.float64)
    normal = _normalize(np.cross(image_axis, depth_axis))
    if normal is None:
        raise ValueError("Sector plane normal is undefined.")
    return normal


def project_point_to_sector(
    point_lps: np.ndarray | list[float],
    pose: WebNavigationPose,
    *,
    max_depth_mm: float,
    sector_angle_deg: float,
    slab_half_thickness_mm: float = DEFAULT_SECTOR_SLAB_HALF_THICKNESS_MM,
) -> dict[str, float | bool]:
    point = np.asarray(point_lps, dtype=np.float64)
    origin = np.asarray(pose.position_lps, dtype=np.float64)
    image_axis = cephalic_image_axis_lps(pose)
    plane_normal = sector_plane_normal_lps(pose)
    depth_axis = np.asarray(pose.depth_axis_lps, dtype=np.float64)

    offset = point - origin
    depth_mm = float(np.dot(offset, depth_axis))
    lateral_mm = float(np.dot(offset, image_axis))
    out_of_plane_mm = float(np.dot(offset, plane_normal))
    half_width_at_depth = max(0.0, depth_mm) * math.tan(math.radians(sector_angle_deg / 2.0))
    visible = bool(
        0.0 <= depth_mm <= max_depth_mm
        and abs(lateral_mm) <= (half_width_at_depth + 1e-9)
        and abs(out_of_plane_mm) <= slab_half_thickness_mm
    )
    return {
        "depth_mm": depth_mm,
        "lateral_mm": lateral_mm,
        "image_x_mm": lateral_mm,
        "out_of_plane_mm": out_of_plane_mm,
        "visible": visible,
        "normalized_depth": float(depth_mm / max_depth_mm) if max_depth_mm > EPSILON else 0.0,
        "normalized_lateral": float(lateral_mm / max_depth_mm) if max_depth_mm > EPSILON else 0.0,
        "slab_half_thickness_mm": float(slab_half_thickness_mm),
    }


def build_navigation_response(
    *,
    centerlines_by_index: Mapping[int, CenterlinePolyline],
    preset: WebPresetNavigation | None,
    line_index: int,
    centerline_s_mm: float,
    roll_deg: float,
    max_depth_mm: float,
    sector_angle_deg: float,
) -> dict[str, object]:
    if line_index not in centerlines_by_index:
        raise ValueError(f"Centerline line_index {line_index} is not available.")

    target_lps = None if preset is None else np.asarray(preset.target_lps, dtype=np.float64)
    pose = navigation_pose_from_polyline(
        centerlines_by_index[line_index],
        centerline_s_mm=centerline_s_mm,
        roll_deg=roll_deg,
        target_lps=target_lps,
        contact_lps=(None if preset is None else np.asarray(preset.contact_lps, dtype=np.float64)),
        contact_centerline_s_mm=(None if preset is None or int(preset.line_index) != int(line_index) else float(preset.centerline_s_mm)),
        shaft_axis_lps=(None if preset is None or preset.shaft_axis_lps is None else np.asarray(preset.shaft_axis_lps, dtype=np.float64)),
        depth_axis_lps=(None if preset is None or preset.depth_axis_lps is None else np.asarray(preset.depth_axis_lps, dtype=np.float64)),
    )

    sector_labels: list[dict[str, object]] = []
    if preset is not None:
        projected_target = project_point_to_sector(
            preset.target_lps,
            pose,
            max_depth_mm=max_depth_mm,
            sector_angle_deg=sector_angle_deg,
        )
        sector_labels.append(
            {
                "id": f"{preset.preset_key}:lymph_node",
                "kind": "lymph_node",
                "label": f"Station {preset.station.upper()} lymph node",
                **projected_target,
            }
        )

    return {
        "pose": {
            "line_index": pose.line_index,
            "centerline_s_mm": pose.centerline_s_mm,
            "position_lps": pose.position_lps,
            "position": lps_to_web(pose.position_lps),
            "tangent_lps": pose.tangent_lps,
            "tangent": lps_to_web(pose.tangent_lps),
            "depth_axis_lps": pose.depth_axis_lps,
            "depth_axis": lps_to_web(pose.depth_axis_lps),
            "lateral_axis_lps": pose.lateral_axis_lps,
            "lateral_axis": lps_to_web(pose.lateral_axis_lps),
            "cephalic_image_axis_lps": [float(value) for value in cephalic_image_axis_lps(pose).tolist()],
            "cephalic_image_axis": lps_to_web(cephalic_image_axis_lps(pose)),
            "roll_deg": pose.roll_deg,
        },
        "sector": {
            "max_depth_mm": float(max_depth_mm),
            "sector_angle_deg": float(sector_angle_deg),
            "labels": sector_labels,
        },
    }
