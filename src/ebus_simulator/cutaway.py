from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ebus_simulator.models import PolyData

EPSILON = 1e-8
PATIENT_LEFT_AXIS_LPS = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


@dataclass(slots=True)
class CutawayDisplay:
    mode: str
    side: str
    side_source: str
    open_side: str
    origin_mode: str
    reference_origin_world: np.ndarray
    origin_world: np.ndarray
    normal_world: np.ndarray
    depth_mm: float
    show_full_airway: bool
    slab_thickness_mm: float | None
    mesh_source: str
    triangles_world: np.ndarray
    warnings: list[str] = field(default_factory=list)


def default_cutaway_side(station: str, approach: str | None) -> str:
    normalized_station = station.strip().lower()
    normalized_approach = None if approach is None else approach.strip().lower()
    if normalized_station in {"4r", "10r", "11ri", "11rs"}:
        return "right"
    if normalized_station in {"4l", "11l"}:
        return "left"
    if normalized_station == "7":
        if normalized_approach == "lms":
            return "left"
        if normalized_approach == "rms":
            return "right"
    return "auto"


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        raise ValueError("Cutaway axis is undefined.")
    return vector / norm


def _resolve_reference_origin(
    *,
    origin_mode: str,
    contact_world: np.ndarray,
    probe_origin_world: np.ndarray | None,
    custom_origin_world: np.ndarray | None,
) -> np.ndarray:
    if origin_mode == "contact":
        return np.asarray(contact_world, dtype=np.float64)
    if origin_mode == "probe_origin":
        if probe_origin_world is None:
            raise ValueError("cutaway_origin='probe_origin' requires a probe origin.")
        return np.asarray(probe_origin_world, dtype=np.float64)
    if origin_mode == "custom":
        if custom_origin_world is None:
            raise ValueError("cutaway_origin='custom' requires a custom origin point.")
        return np.asarray(custom_origin_world, dtype=np.float64)
    raise ValueError(f"Unsupported cutaway origin mode: {origin_mode}")


def _resolve_side(requested_side: str, *, station: str, approach: str | None) -> tuple[str, str]:
    normalized_side = requested_side.strip().lower()
    if normalized_side in {"left", "right"}:
        return normalized_side, "cli"
    if normalized_side != "auto":
        raise ValueError(f"Unsupported cutaway side: {requested_side}")
    default_side = default_cutaway_side(station, approach)
    if default_side == "auto":
        return "auto", "visibility_auto"
    return default_side, "station_rule"


def _visibility_score(
    normal_world: np.ndarray,
    *,
    reference_origin_world: np.ndarray,
    target_world: np.ndarray,
    station_visibility_points_world: np.ndarray | None,
) -> float:
    target_score = max(0.0, float(np.dot(target_world - reference_origin_world, normal_world)))
    score = 4.0 * target_score
    if station_visibility_points_world is not None and station_visibility_points_world.size:
        station_dots = (station_visibility_points_world - reference_origin_world[None, :]) @ normal_world
        positive = np.maximum(station_dots, 0.0)
        if positive.size:
            score += float(np.mean(positive))
    return score


def _choose_normal_sign(
    base_normal_world: np.ndarray,
    *,
    side: str,
    reference_origin_world: np.ndarray,
    target_world: np.ndarray,
    station_visibility_points_world: np.ndarray | None,
    warnings: list[str],
) -> np.ndarray:
    normal_world = _normalize(base_normal_world)
    lateral_component = float(np.dot(normal_world, PATIENT_LEFT_AXIS_LPS))

    if side == "left" and abs(lateral_component) > 1e-4:
        return normal_world if lateral_component > 0.0 else (-normal_world)
    if side == "right" and abs(lateral_component) > 1e-4:
        return normal_world if lateral_component < 0.0 else (-normal_world)
    if side in {"left", "right"} and abs(lateral_component) <= 1e-4:
        warnings.append(
            f"Cutaway side '{side}' could not be oriented from the LPS left/right axis alone; visibility fallback chose the normal sign."
        )

    score_positive = _visibility_score(
        normal_world,
        reference_origin_world=reference_origin_world,
        target_world=target_world,
        station_visibility_points_world=station_visibility_points_world,
    )
    score_negative = _visibility_score(
        -normal_world,
        reference_origin_world=reference_origin_world,
        target_world=target_world,
        station_visibility_points_world=station_visibility_points_world,
    )
    return normal_world if score_positive >= score_negative else (-normal_world)


def _infer_open_side(normal_world: np.ndarray) -> str:
    lateral_component = float(np.dot(normal_world, PATIENT_LEFT_AXIS_LPS))
    if lateral_component > 1e-4:
        return "left"
    if lateral_component < -1e-4:
        return "right"
    return "center"


def _clip_polygon_half_space(vertices: np.ndarray, signed_distances: np.ndarray) -> np.ndarray:
    if vertices.shape[0] == 0:
        return vertices

    output: list[np.ndarray] = []
    for index in range(vertices.shape[0]):
        current = vertices[index]
        next_point = vertices[(index + 1) % vertices.shape[0]]
        current_distance = float(signed_distances[index])
        next_distance = float(signed_distances[(index + 1) % vertices.shape[0]])
        current_inside = current_distance <= EPSILON
        next_inside = next_distance <= EPSILON

        if current_inside and next_inside:
            output.append(next_point)
            continue

        if current_inside and not next_inside:
            t = current_distance / (current_distance - next_distance)
            output.append(current + (t * (next_point - current)))
            continue

        if (not current_inside) and next_inside:
            t = current_distance / (current_distance - next_distance)
            output.append(current + (t * (next_point - current)))
            output.append(next_point)

    if not output:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _mesh_to_triangles_world(mesh: PolyData) -> np.ndarray:
    if not mesh.polygons:
        return np.empty((0, 3, 3), dtype=np.float64)

    triangles: list[np.ndarray] = []
    for polygon in mesh.polygons:
        if polygon.size < 3:
            continue
        polygon_points = mesh.points_lps[np.asarray(polygon, dtype=np.int64)]
        anchor = polygon_points[0]
        for index in range(1, polygon_points.shape[0] - 1):
            triangles.append(
                np.stack((anchor, polygon_points[index], polygon_points[index + 1]), axis=0).astype(np.float64)
            )

    if not triangles:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.asarray(triangles, dtype=np.float64)


def _clip_triangles_world(
    triangles_world: np.ndarray,
    *,
    origin_world: np.ndarray,
    normal_world: np.ndarray,
    slab_thickness_mm: float | None,
) -> np.ndarray:
    clipped: list[np.ndarray] = []

    for triangle in triangles_world:
        polygon = np.asarray(triangle, dtype=np.float64)
        signed_distances = (polygon - origin_world[None, :]) @ normal_world
        polygon = _clip_polygon_half_space(polygon, signed_distances)
        if polygon.shape[0] < 3:
            continue

        if slab_thickness_mm is not None and slab_thickness_mm > 0.0:
            slab_distances = -(((polygon - origin_world[None, :]) @ normal_world) + slab_thickness_mm)
            polygon = _clip_polygon_half_space(polygon, slab_distances)
            if polygon.shape[0] < 3:
                continue

        anchor = polygon[0]
        for index in range(1, polygon.shape[0] - 1):
            clipped.append(np.stack((anchor, polygon[index], polygon[index + 1]), axis=0))

    if not clipped:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def build_display_cutaway(
    mesh: PolyData,
    *,
    mesh_source: str,
    station: str,
    approach: str | None,
    mode: str,
    requested_side: str,
    origin_mode: str,
    depth_mm: float,
    show_full_airway: bool,
    contact_world: np.ndarray,
    target_world: np.ndarray,
    lateral_axis_world: np.ndarray,
    probe_axis_world: np.ndarray,
    shaft_axis_world: np.ndarray,
    probe_origin_world: np.ndarray | None = None,
    custom_origin_world: np.ndarray | None = None,
    station_visibility_points_world: np.ndarray | None = None,
    slab_thickness_mm: float | None = None,
) -> CutawayDisplay:
    if depth_mm < 0.0:
        raise ValueError("cutaway_depth_mm must be non-negative.")

    axis_lookup = {
        "lateral": np.asarray(lateral_axis_world, dtype=np.float64),
        "probe_axis": np.asarray(probe_axis_world, dtype=np.float64),
        "shaft_axis": np.asarray(shaft_axis_world, dtype=np.float64),
    }
    normalized_mode = mode.strip().lower()
    if normalized_mode not in axis_lookup:
        raise ValueError(f"Unsupported cutaway mode: {mode}")

    warnings: list[str] = []
    side, side_source = _resolve_side(requested_side, station=station, approach=approach)
    reference_origin_world = _resolve_reference_origin(
        origin_mode=origin_mode,
        contact_world=np.asarray(contact_world, dtype=np.float64),
        probe_origin_world=(None if probe_origin_world is None else np.asarray(probe_origin_world, dtype=np.float64)),
        custom_origin_world=(None if custom_origin_world is None else np.asarray(custom_origin_world, dtype=np.float64)),
    )
    normal_world = _choose_normal_sign(
        axis_lookup[normalized_mode],
        side=side,
        reference_origin_world=reference_origin_world,
        target_world=np.asarray(target_world, dtype=np.float64),
        station_visibility_points_world=station_visibility_points_world,
        warnings=warnings,
    )
    origin_world = reference_origin_world - (depth_mm * normal_world)

    triangles_world = _mesh_to_triangles_world(mesh)
    if show_full_airway:
        clipped_triangles = triangles_world
    else:
        clipped_triangles = _clip_triangles_world(
            triangles_world,
            origin_world=origin_world,
            normal_world=normal_world,
            slab_thickness_mm=slab_thickness_mm,
        )
        if triangles_world.shape[0] and clipped_triangles.shape[0] == 0:
            warnings.append("Cutaway clipping removed the full airway display mesh.")

    return CutawayDisplay(
        mode=normalized_mode,
        side=side,
        side_source=side_source,
        open_side=_infer_open_side(normal_world),
        origin_mode=origin_mode,
        reference_origin_world=reference_origin_world,
        origin_world=origin_world,
        normal_world=normal_world,
        depth_mm=float(depth_mm),
        show_full_airway=bool(show_full_airway),
        slab_thickness_mm=slab_thickness_mm,
        mesh_source=mesh_source,
        triangles_world=clipped_triangles,
        warnings=warnings,
    )
