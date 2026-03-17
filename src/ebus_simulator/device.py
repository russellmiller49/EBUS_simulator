from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ebus_simulator.centerline import CenterlineGraph, CenterlineProjection
from ebus_simulator.geometry import (
    distance_to_mask_surface_mm,
    sample_signed_distance_mm,
    sample_volume_scalar,
)
from ebus_simulator.mesh_geometry import MeshQueryResult, get_mesh_surface
from ebus_simulator.models import PolyData, VolumeData


EPSILON = 1e-9
CONTACT_HU_THRESHOLD = -600.0
CONTACT_SCAN_FORWARD_MM = 8.0
CONTACT_SCAN_BACKWARD_MM = 6.0
CONTACT_SCAN_SHAFT_MM = 2.0
CONTACT_SCAN_STEP_MM = 0.5
CONTACT_AMBIGUITY_SCORE_DELTA = 0.35
MESH_SCAN_FORWARD_MM = 3.0
MESH_SCAN_BACKWARD_MM = 2.0
MESH_SCAN_SHAFT_MM = 1.5
MESH_SCAN_STEP_MM = 0.5
NORMAL_ORIENTATION_DELTA_MM = 0.75


@dataclass(frozen=True, slots=True)
class CPEBUSDeviceModel:
    name: str
    shaft_label: str
    video_axis_offset_deg: float
    probe_origin_offset_mm: float
    sector_angle_deg: float
    displayed_range_mm: float
    source_oblique_size_mm: float
    reference_fov_mm: float


@dataclass(slots=True)
class ContactRefinement:
    original_contact_world: list[float]
    voxel_refined_contact_world: list[float]
    refined_contact_world: list[float]
    original_contact_to_airway_distance_mm: float
    voxel_refined_contact_to_airway_distance_mm: float
    refined_contact_to_airway_distance_mm: float
    voxel_to_mesh_contact_distance_mm: float
    refinement_applied: bool
    refinement_method: str
    voxel_refinement_method: str
    mesh_refinement_method: str
    candidate_hu: float | None
    candidate_branch_graph_name: str | None
    candidate_branch_line_index: int | None
    branch_hint: str | None
    branch_hint_applied: bool
    branch_hint_match: bool | None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BranchHintSpec:
    raw_value: str
    any_lines: frozenset[int] = frozenset()
    main_lines: frozenset[int] = frozenset()
    network_lines: frozenset[int] = frozenset()

    def matches(self, projection: CenterlineProjection) -> bool:
        if int(projection.line_index) in self.any_lines:
            return True
        if projection.graph_name == "main" and int(projection.line_index) in self.main_lines:
            return True
        if projection.graph_name == "network" and int(projection.line_index) in self.network_lines:
            return True
        return False


@dataclass(slots=True)
class DevicePose:
    device_model: CPEBUSDeviceModel
    tip_start_world: list[float]
    probe_origin_world: list[float]
    shaft_axis_world: list[float]
    video_axis_world: list[float]
    probe_axis_world: list[float]
    lateral_axis_world: list[float]
    wall_normal_world: list[float]
    voxel_wall_normal_world: list[float]
    voxel_probe_axis_world: list[float]
    target_world: list[float]
    contact_refinement: ContactRefinement


def get_cp_ebus_device_model(name: str) -> CPEBUSDeviceModel:
    normalized = name.strip().lower()
    if normalized != "bf_uc180f":
        raise ValueError(f"Unsupported CP-EBUS device model {name!r}. Expected 'bf_uc180f'.")
    return CPEBUSDeviceModel(
        name="bf_uc180f",
        shaft_label="Olympus BF-UC180F-like",
        video_axis_offset_deg=20.0,
        probe_origin_offset_mm=6.0,
        sector_angle_deg=60.0,
        displayed_range_mm=40.0,
        source_oblique_size_mm=51.79,
        reference_fov_mm=100.0,
    )


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) - (float(np.dot(vector, axis)) * np.asarray(axis, dtype=np.float64))


def _surface_distance(point_lps: np.ndarray, airway_lumen: VolumeData, airway_solid: VolumeData) -> float:
    return min(
        distance_to_mask_surface_mm(point_lps, airway_lumen),
        distance_to_mask_surface_mm(point_lps, airway_solid),
    )


def _rotation_toward(base_axis: np.ndarray, toward_axis: np.ndarray, offset_deg: float) -> np.ndarray:
    angle = math.radians(offset_deg)
    rotated = (math.cos(angle) * base_axis) + (math.sin(angle) * toward_axis)
    normalized = _normalize(rotated)
    if normalized is None:
        raise ValueError("Failed to construct video axis for CP-EBUS device model.")
    return normalized


def _rotate_around_axis(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) <= EPSILON:
        return np.asarray(vector, dtype=np.float64)
    axis_unit = _normalize(axis)
    if axis_unit is None:
        return np.asarray(vector, dtype=np.float64)
    angle = math.radians(angle_deg)
    return (
        vector * math.cos(angle)
        + np.cross(axis_unit, vector) * math.sin(angle)
        + axis_unit * np.dot(axis_unit, vector) * (1.0 - math.cos(angle))
    )


def _fallback_wall_normal(
    contact_world: np.ndarray,
    shaft_axis: np.ndarray,
    projection: CenterlineProjection | None,
    fallback_probe_axis: np.ndarray,
) -> np.ndarray:
    if projection is not None:
        tangent = projection.tangent_lps if projection.tangent_lps is not None else shaft_axis
        radial = _project_perpendicular(contact_world - projection.closest_point_lps, tangent)
        radial_normalized = _normalize(radial)
        if radial_normalized is not None:
            return radial_normalized
    return np.asarray(fallback_probe_axis, dtype=np.float64)


def _candidate_branch_projection(
    point_lps: np.ndarray,
    *,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
    branch_hint: str | None = None,
) -> CenterlineProjection | None:
    hint_spec = _parse_branch_hint(branch_hint)
    projections = _candidate_branch_projections(point_lps, main_graph=main_graph, network_graph=network_graph)
    if hint_spec is not None:
        matching = [projection for projection in projections if hint_spec.matches(projection)]
        if matching:
            return min(matching, key=lambda projection: projection.distance_mm)
    if projections:
        main_projection = next((projection for projection in projections if projection.graph_name == "main"), None)
        if main_projection is not None:
            return main_projection
        return min(projections, key=lambda projection: projection.distance_mm)
    return None


def _candidate_branch_projections(
    point_lps: np.ndarray,
    *,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
) -> list[CenterlineProjection]:
    projections: list[CenterlineProjection] = []
    main_projection = main_graph.nearest_point(point_lps)
    if main_projection is not None:
        projections.append(main_projection)
    if network_graph is not None:
        network_projection = network_graph.nearest_point(point_lps)
        if network_projection is not None:
            projections.append(network_projection)
    return projections


def _parse_branch_hint(branch_hint: str | None) -> BranchHintSpec | None:
    if branch_hint is None:
        return None

    any_lines: set[int] = set()
    main_lines: set[int] = set()
    network_lines: set[int] = set()
    for raw_token in branch_hint.replace(";", ",").split(","):
        token = raw_token.strip().lower()
        if not token:
            continue

        prefix = "line"
        value = token
        for separator in (":", "="):
            if separator in token:
                prefix, value = token.split(separator, 1)
                prefix = prefix.strip()
                value = value.strip()
                break

        if prefix in {"line", "branch"}:
            any_lines.add(int(value))
            continue
        if prefix in {"main", "main_line"}:
            main_lines.add(int(value))
            continue
        if prefix in {"network", "network_line"}:
            network_lines.add(int(value))
            continue
        raise ValueError(
            f"Unsupported branch_hint token {raw_token!r}. Expected line, main:<index>, or network:<index>."
        )

    if not any_lines and not main_lines and not network_lines:
        return None
    return BranchHintSpec(
        raw_value=str(branch_hint),
        any_lines=frozenset(any_lines),
        main_lines=frozenset(main_lines),
        network_lines=frozenset(network_lines),
    )


def _projection_key(projection: CenterlineProjection | None) -> tuple[str, int] | None:
    if projection is None:
        return None
    return projection.graph_name, int(projection.line_index)


def _build_probe_axis(
    *,
    contact_world: np.ndarray,
    target_world: np.ndarray,
    shaft_axis: np.ndarray,
    wall_normal: np.ndarray,
    fallback_probe_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projected_wall_normal = _project_perpendicular(wall_normal, shaft_axis)
    probe_axis = _normalize(projected_wall_normal)
    if probe_axis is None:
        return np.asarray(fallback_probe_axis, dtype=np.float64), projected_wall_normal
    if float(np.dot(target_world - contact_world, probe_axis)) < 0.0:
        probe_axis = -probe_axis
    return probe_axis, projected_wall_normal


def _orient_mesh_normal(
    normal_world: np.ndarray,
    *,
    contact_world: np.ndarray,
    airway_lumen: VolumeData,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
    shaft_axis: np.ndarray,
    fallback_probe_axis: np.ndarray,
    branch_hint: str | None = None,
) -> np.ndarray:
    oriented = _normalize(normal_world)
    if oriented is None:
        return _fallback_wall_normal(
            contact_world,
            shaft_axis,
            _candidate_branch_projection(
                contact_world,
                main_graph=main_graph,
                network_graph=network_graph,
                branch_hint=branch_hint,
            ),
            fallback_probe_axis,
        )

    forward_sd = sample_signed_distance_mm(contact_world + (NORMAL_ORIENTATION_DELTA_MM * oriented), airway_lumen)
    backward_sd = sample_signed_distance_mm(contact_world - (NORMAL_ORIENTATION_DELTA_MM * oriented), airway_lumen)
    if forward_sd < backward_sd:
        oriented = -oriented

    projection = _candidate_branch_projection(
        contact_world,
        main_graph=main_graph,
        network_graph=network_graph,
        branch_hint=branch_hint,
    )
    if projection is not None:
        tangent = projection.tangent_lps if projection.tangent_lps is not None else shaft_axis
        radial = _project_perpendicular(contact_world - projection.closest_point_lps, tangent)
        radial_unit = _normalize(radial)
        if radial_unit is not None and float(np.dot(oriented, radial_unit)) < 0.0:
            oriented = -oriented

    return oriented


def _estimate_voxel_wall_normal(point_lps: np.ndarray, airway_solid: VolumeData) -> np.ndarray | None:
    gradient = np.zeros(3, dtype=np.float64)
    for axis_index in range(3):
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = 1.0
        forward = sample_signed_distance_mm(point_lps + (axis * NORMAL_ORIENTATION_DELTA_MM), airway_solid)
        backward = sample_signed_distance_mm(point_lps - (axis * NORMAL_ORIENTATION_DELTA_MM), airway_solid)
        gradient[axis_index] = (forward - backward) / (2.0 * NORMAL_ORIENTATION_DELTA_MM)
    return _normalize(gradient)


def _apply_axis_sign_override(
    probe_axis: np.ndarray,
    lateral_axis: np.ndarray,
    *,
    shaft_axis: np.ndarray,
    axis_sign_override: str | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    if axis_sign_override is None:
        return probe_axis, lateral_axis, None

    normalized_override = axis_sign_override.strip().lower()
    if normalized_override in {"flip_probe_axis", "flip_nus", "probe_axis", "nus"}:
        adjusted_probe = -probe_axis
        adjusted_lateral = _normalize(np.cross(shaft_axis, adjusted_probe))
        if adjusted_lateral is None:
            return probe_axis, lateral_axis, f"Axis-sign override {axis_sign_override!r} produced an invalid lateral axis and was ignored."
        return adjusted_probe, adjusted_lateral, None

    if normalized_override in {"flip_lateral_axis", "flip_lateral", "lateral_axis", "lateral"}:
        return probe_axis, -lateral_axis, None

    if normalized_override in {"flip_both", "both"}:
        adjusted_probe = -probe_axis
        adjusted_lateral = _normalize(np.cross(shaft_axis, adjusted_probe))
        if adjusted_lateral is None:
            return probe_axis, lateral_axis, f"Axis-sign override {axis_sign_override!r} produced an invalid orthonormal frame and was ignored."
        return adjusted_probe, adjusted_lateral, None

    return probe_axis, lateral_axis, f"Axis-sign override {axis_sign_override!r} is unsupported and was ignored."


def _refine_airway_contact_voxel(
    seed_contact_world: np.ndarray,
    *,
    shaft_axis: np.ndarray,
    probe_axis: np.ndarray,
    ct_volume: VolumeData,
    airway_lumen: VolumeData,
    airway_solid: VolumeData,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None = None,
    branch_hint: str | None = None,
) -> tuple[np.ndarray, float, str, float | None, str | None, int | None, list[str]]:
    original_surface_distance = _surface_distance(seed_contact_world, airway_lumen, airway_solid)
    warnings: list[str] = []

    seed_projection = _candidate_branch_projection(
        seed_contact_world,
        main_graph=main_graph,
        network_graph=network_graph,
        branch_hint=branch_hint,
    )
    seed_branch_key = _projection_key(seed_projection)

    offsets_us = np.arange(-CONTACT_SCAN_BACKWARD_MM, CONTACT_SCAN_FORWARD_MM + CONTACT_SCAN_STEP_MM, CONTACT_SCAN_STEP_MM, dtype=np.float64)
    offsets_b = np.arange(-CONTACT_SCAN_SHAFT_MM, CONTACT_SCAN_SHAFT_MM + CONTACT_SCAN_STEP_MM, CONTACT_SCAN_STEP_MM, dtype=np.float64)
    candidates: list[tuple[float, np.ndarray, float, str | None, int | None]] = []

    for offset_b in offsets_b:
        for offset_us in offsets_us:
            point_lps = seed_contact_world + (offset_us * probe_axis) + (offset_b * shaft_axis)
            hu_value = sample_volume_scalar(point_lps, ct_volume, order=1, cval=-1000.0)
            if hu_value <= CONTACT_HU_THRESHOLD:
                continue

            projection = _candidate_branch_projection(
                point_lps,
                main_graph=main_graph,
                network_graph=network_graph,
                branch_hint=branch_hint,
            )
            surface_distance = _surface_distance(point_lps, airway_lumen, airway_solid)
            branch_penalty = 0.0
            branch_graph_name = None
            branch_index = None
            direction_penalty = 0.0

            if projection is not None:
                branch_graph_name = projection.graph_name
                branch_index = int(projection.line_index)
                if seed_branch_key is not None and _projection_key(projection) != seed_branch_key:
                    branch_penalty = 1.5
                radial_vector = point_lps - projection.closest_point_lps
                tangent = projection.tangent_lps if projection.tangent_lps is not None else shaft_axis
                radial_unit = _normalize(_project_perpendicular(radial_vector, tangent))
                if radial_unit is not None:
                    direction_penalty = max(0.0, 0.75 - float(np.dot(radial_unit, probe_axis)))

            score = surface_distance + (0.10 * abs(float(offset_us))) + (0.15 * abs(float(offset_b))) + branch_penalty + direction_penalty
            candidates.append((score, point_lps, hu_value, branch_graph_name, branch_index))
            break

    if not candidates:
        warnings.append("Voxel contact refinement found no tissue candidate above the HU threshold; the markup contact was retained.")
        seed_branch_graph_name = None if seed_projection is None else seed_projection.graph_name
        seed_branch_index = None if seed_projection is None else int(seed_projection.line_index)
        return (
            seed_contact_world.copy(),
            float(original_surface_distance),
            "seed_fallback",
            None,
            seed_branch_graph_name,
            seed_branch_index,
            warnings,
        )

    candidates.sort(key=lambda item: item[0])
    best_score, best_point, best_hu, best_branch_graph_name, best_branch = candidates[0]
    refined_surface_distance = _surface_distance(best_point, airway_lumen, airway_solid)

    if len(candidates) > 1:
        second_score, second_point, _, second_branch_graph_name, second_branch = candidates[1]
        separation = float(np.linalg.norm(second_point - best_point))
        second_branch_key = (
            None
            if second_branch is None or second_branch_graph_name is None
            else (second_branch_graph_name, int(second_branch))
        )
        best_branch_key = (
            None
            if best_branch is None or best_branch_graph_name is None
            else (best_branch_graph_name, int(best_branch))
        )
        same_branch_ambiguity = best_branch_key is None or second_branch_key is None or best_branch_key == second_branch_key
        if abs(second_score - best_score) <= CONTACT_AMBIGUITY_SCORE_DELTA and separation > 1.0 and same_branch_ambiguity:
            warnings.append("Voxel contact refinement was ambiguous; the best airway-wall candidate was chosen deterministically.")

    if refined_surface_distance > (original_surface_distance + 0.75):
        warnings.append("Voxel contact refinement did not improve airway-wall proximity; the markup contact was retained.")
        return (
            seed_contact_world.copy(),
            float(original_surface_distance),
            "seed_retained",
            float(best_hu),
            best_branch_graph_name,
            best_branch,
            warnings,
        )

    if refined_surface_distance > 2.0:
        warnings.append(f"Voxel-refined contact remains {refined_surface_distance:.2f} mm from the airway wall.")

    return (
        np.asarray(best_point, dtype=np.float64),
        float(refined_surface_distance),
        "ct_threshold_wall_search",
        float(best_hu),
        best_branch_graph_name,
        best_branch,
        warnings,
    )


def _score_mesh_candidate(
    *,
    seed_point: np.ndarray,
    mesh_query: MeshQueryResult,
    mesh_contact_world: np.ndarray,
    mesh_normal_world: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    seed_branch_key: tuple[str, int] | None,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
    branch_hint: str | None,
    offset_us: float,
    offset_b: float,
) -> tuple[float, str | None, int | None]:
    projection = _candidate_branch_projection(
        mesh_contact_world,
        main_graph=main_graph,
        network_graph=network_graph,
        branch_hint=branch_hint,
    )
    branch_graph_name = None
    branch_index = None
    branch_penalty = 0.0
    radial_penalty = 0.0
    centerline_penalty = 0.0
    normal_penalty = 0.0

    if projection is not None:
        branch_graph_name = projection.graph_name
        branch_index = int(projection.line_index)
        if seed_branch_key is not None and _projection_key(projection) != seed_branch_key:
            branch_penalty = 1.5
        tangent = projection.tangent_lps if projection.tangent_lps is not None else shaft_axis
        radial_vector = _project_perpendicular(mesh_contact_world - projection.closest_point_lps, tangent)
        radial_unit = _normalize(radial_vector)
        if radial_unit is not None:
            radial_penalty = max(0.0, 0.65 - float(np.dot(radial_unit, probe_axis)))
        centerline_penalty = 0.05 * float(projection.distance_mm)

    normal_penalty = max(0.0, 0.50 - float(np.dot(mesh_normal_world, probe_axis)))

    return (
        float(mesh_query.distance_mm)
        + (0.20 * float(np.linalg.norm(mesh_contact_world - seed_point)))
        + (0.10 * abs(float(offset_us)))
        + (0.15 * abs(float(offset_b)))
        + branch_penalty
        + radial_penalty
        + centerline_penalty
        + normal_penalty,
        branch_graph_name,
        branch_index,
    )


def _refine_airway_contact_mesh(
    seed_contact_world: np.ndarray,
    *,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    raw_airway_mesh: PolyData,
    airway_lumen: VolumeData,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
    fallback_probe_axis: np.ndarray,
    branch_hint: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, str, str | None, int | None, list[str]]:
    surface = get_mesh_surface(raw_airway_mesh)
    warnings: list[str] = []
    seed_projection = _candidate_branch_projection(
        seed_contact_world,
        main_graph=main_graph,
        network_graph=network_graph,
        branch_hint=branch_hint,
    )
    seed_branch_key = _projection_key(seed_projection)

    offsets_us = np.arange(-MESH_SCAN_BACKWARD_MM, MESH_SCAN_FORWARD_MM + MESH_SCAN_STEP_MM, MESH_SCAN_STEP_MM, dtype=np.float64)
    offsets_b = np.arange(-MESH_SCAN_SHAFT_MM, MESH_SCAN_SHAFT_MM + MESH_SCAN_STEP_MM, MESH_SCAN_STEP_MM, dtype=np.float64)
    candidates: list[tuple[float, np.ndarray, np.ndarray, float, str | None, int | None]] = []

    for offset_b in offsets_b:
        for offset_us in offsets_us:
            sample_point = seed_contact_world + (offset_us * probe_axis) + (offset_b * shaft_axis)
            query = surface.nearest_point(sample_point)
            normal = query.point_normal_lps if query.point_normal_lps is not None else query.face_normal_lps
            if normal is None:
                continue
            oriented_normal = _orient_mesh_normal(
                normal,
                contact_world=query.closest_point_lps,
                airway_lumen=airway_lumen,
                main_graph=main_graph,
                network_graph=network_graph,
                shaft_axis=shaft_axis,
                fallback_probe_axis=fallback_probe_axis,
                branch_hint=branch_hint,
            )
            score, branch_graph_name, branch_index = _score_mesh_candidate(
                seed_point=sample_point,
                mesh_query=query,
                mesh_contact_world=query.closest_point_lps,
                mesh_normal_world=oriented_normal,
                probe_axis=probe_axis,
                shaft_axis=shaft_axis,
                seed_branch_key=seed_branch_key,
                main_graph=main_graph,
                network_graph=network_graph,
                branch_hint=branch_hint,
                offset_us=float(offset_us),
                offset_b=float(offset_b),
            )
            candidates.append((score, query.closest_point_lps, oriented_normal, float(query.distance_mm), branch_graph_name, branch_index))

    if not candidates:
        query = surface.nearest_point(seed_contact_world)
        normal = query.point_normal_lps if query.point_normal_lps is not None else query.face_normal_lps
        if normal is None:
            warnings.append("Mesh contact refinement found no valid surface normal; the seed contact was retained.")
            seed_branch_graph_name = None if seed_projection is None else seed_projection.graph_name
            seed_branch_index = None if seed_projection is None else int(seed_projection.line_index)
            return (
                seed_contact_world.copy(),
                np.asarray(fallback_probe_axis, dtype=np.float64),
                float("inf"),
                "mesh_seed_fallback",
                seed_branch_graph_name,
                seed_branch_index,
                warnings,
            )
        oriented_normal = _orient_mesh_normal(
            normal,
            contact_world=query.closest_point_lps,
            airway_lumen=airway_lumen,
            main_graph=main_graph,
            network_graph=network_graph,
            shaft_axis=shaft_axis,
            fallback_probe_axis=fallback_probe_axis,
            branch_hint=branch_hint,
        )
        warnings.append("Mesh contact refinement fell back to the nearest raw-mesh projection from the seed contact.")
        seed_branch_graph_name = None if seed_projection is None else seed_projection.graph_name
        seed_branch_index = None if seed_projection is None else int(seed_projection.line_index)
        return (
            np.asarray(query.closest_point_lps, dtype=np.float64),
            oriented_normal,
            float(query.distance_mm),
            "mesh_nearest_projection",
            seed_branch_graph_name,
            seed_branch_index,
            warnings,
        )

    candidates.sort(key=lambda item: item[0])
    best_score, best_contact, best_normal, best_distance, best_branch_graph_name, best_branch = candidates[0]
    if len(candidates) > 1:
        second_score, second_contact, _, _, second_branch_graph_name, second_branch = candidates[1]
        separation = float(np.linalg.norm(second_contact - best_contact))
        second_branch_key = (
            None
            if second_branch is None or second_branch_graph_name is None
            else (second_branch_graph_name, int(second_branch))
        )
        best_branch_key = (
            None
            if best_branch is None or best_branch_graph_name is None
            else (best_branch_graph_name, int(best_branch))
        )
        same_branch_ambiguity = best_branch_key is None or second_branch_key is None or best_branch_key == second_branch_key
        if abs(second_score - best_score) <= CONTACT_AMBIGUITY_SCORE_DELTA and separation > 1.0 and same_branch_ambiguity:
            warnings.append("Mesh contact refinement was ambiguous; the best raw-mesh candidate was chosen deterministically.")

    if best_distance > 2.0:
        warnings.append(f"Mesh-refined contact remains {best_distance:.2f} mm from the raw airway mesh.")

    return (
        np.asarray(best_contact, dtype=np.float64),
        np.asarray(best_normal, dtype=np.float64),
        float(best_distance),
        "raw_mesh_projection_search",
        best_branch_graph_name,
        best_branch,
        warnings,
    )


def build_device_pose(
    pose,
    *,
    device_name: str,
    ct_volume: VolumeData,
    airway_lumen: VolumeData,
    airway_solid: VolumeData,
    raw_airway_mesh: PolyData | None,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None = None,
    refine_contact: bool = True,
    roll_offset_deg: float = 0.0,
    axis_sign_override: str | None = None,
    branch_hint: str | None = None,
    contact_seed_world: np.ndarray | None = None,
    shaft_axis_override: np.ndarray | None = None,
    depth_axis_override: np.ndarray | None = None,
) -> DevicePose:
    model = get_cp_ebus_device_model(device_name)

    original_contact = np.asarray(pose.contact_world, dtype=np.float64)
    seed_contact = original_contact if contact_seed_world is None else np.asarray(contact_seed_world, dtype=np.float64)
    target_world = np.asarray(pose.target_world, dtype=np.float64)
    shaft_axis_source = pose.shaft_axis if shaft_axis_override is None else shaft_axis_override
    shaft_axis = _normalize(np.asarray(shaft_axis_source, dtype=np.float64))
    if shaft_axis is None:
        raise ValueError(f"Preset {pose.preset_id!r} approach {pose.contact_approach!r} does not have a valid shaft axis.")

    depth_axis_source = (
        (pose.depth_axis if pose.depth_axis is not None else pose.default_depth_axis)
        if depth_axis_override is None
        else depth_axis_override
    )
    fallback_probe_axis = _normalize(np.asarray(depth_axis_source, dtype=np.float64))
    if fallback_probe_axis is None:
        raise ValueError(f"Preset {pose.preset_id!r} approach {pose.contact_approach!r} does not have a valid depth/probe axis.")

    original_surface_distance = _surface_distance(original_contact, airway_lumen, airway_solid)

    if refine_contact:
        (
            voxel_contact,
            voxel_distance,
            voxel_method,
            candidate_hu,
            voxel_branch_graph_name,
            voxel_branch,
            refinement_warnings,
        ) = _refine_airway_contact_voxel(
            seed_contact,
            shaft_axis=shaft_axis,
            probe_axis=fallback_probe_axis,
            ct_volume=ct_volume,
            airway_lumen=airway_lumen,
            airway_solid=airway_solid,
            main_graph=main_graph,
            network_graph=network_graph,
            branch_hint=branch_hint,
        )
    else:
        voxel_contact = seed_contact.copy()
        voxel_distance = original_surface_distance
        voxel_method = "disabled"
        candidate_hu = None
        voxel_branch_graph_name = None if pose.centerline_query is None else pose.centerline_query.graph_name
        voxel_branch = None if pose.centerline_query is None else pose.centerline_query.line_index
        refinement_warnings = []

    voxel_projection = _candidate_branch_projection(
        voxel_contact,
        main_graph=main_graph,
        network_graph=network_graph,
        branch_hint=branch_hint,
    )
    voxel_wall_normal = _estimate_voxel_wall_normal(voxel_contact, airway_solid)
    if voxel_wall_normal is None:
        voxel_wall_normal = _fallback_wall_normal(voxel_contact, shaft_axis, voxel_projection, fallback_probe_axis)
        refinement_warnings.append("Voxel wall-normal estimation was degenerate; a centerline-radial fallback was used.")

    voxel_probe_axis, voxel_projected_wall_normal = _build_probe_axis(
        contact_world=voxel_contact,
        target_world=target_world,
        shaft_axis=shaft_axis,
        wall_normal=voxel_wall_normal,
        fallback_probe_axis=fallback_probe_axis,
    )
    if _normalize(voxel_projected_wall_normal) is None:
        refinement_warnings.append("Voxel wall-normal projection was degenerate; the fallback probe axis was used.")

    if raw_airway_mesh is not None:
        (
            mesh_contact,
            mesh_wall_normal,
            mesh_distance,
            mesh_method,
            mesh_branch_graph_name,
            mesh_branch,
            mesh_warnings,
        ) = _refine_airway_contact_mesh(
            voxel_contact if refine_contact else seed_contact,
            probe_axis=voxel_probe_axis,
            shaft_axis=shaft_axis,
            raw_airway_mesh=raw_airway_mesh,
            airway_lumen=airway_lumen,
            main_graph=main_graph,
            network_graph=network_graph,
            fallback_probe_axis=fallback_probe_axis,
            branch_hint=branch_hint,
        )
        refinement_warnings.extend(mesh_warnings)
    else:
        mesh_contact = voxel_contact.copy()
        mesh_wall_normal = voxel_wall_normal.copy()
        mesh_distance = float("nan")
        mesh_method = "mesh_unavailable"
        mesh_branch_graph_name = voxel_branch_graph_name
        mesh_branch = voxel_branch
        refinement_warnings.append("Raw airway mesh was unavailable; the voxel-derived contact and wall normal were retained.")

    probe_axis, mesh_projected_wall_normal = _build_probe_axis(
        contact_world=mesh_contact,
        target_world=target_world,
        shaft_axis=shaft_axis,
        wall_normal=mesh_wall_normal,
        fallback_probe_axis=voxel_probe_axis,
    )
    if _normalize(mesh_projected_wall_normal) is None:
        refinement_warnings.append("Mesh wall-normal projection was degenerate; the voxel probe axis was used.")

    if abs(float(roll_offset_deg)) > EPSILON:
        rotated_probe_axis = _normalize(_rotate_around_axis(probe_axis, shaft_axis, float(roll_offset_deg)))
        if rotated_probe_axis is None:
            refinement_warnings.append("Final probe-axis roll override was invalid and was ignored.")
        else:
            probe_axis = rotated_probe_axis

    lateral_axis = _normalize(np.cross(shaft_axis, probe_axis))
    if lateral_axis is None:
        raise ValueError("Failed to construct the lateral axis for the CP-EBUS device pose.")

    probe_axis, lateral_axis, axis_override_warning = _apply_axis_sign_override(
        probe_axis,
        lateral_axis,
        shaft_axis=shaft_axis,
        axis_sign_override=axis_sign_override,
    )
    if axis_override_warning is not None:
        refinement_warnings.append(axis_override_warning)

    video_axis = _rotation_toward(shaft_axis, probe_axis, model.video_axis_offset_deg)
    tip_start_world = mesh_contact - (shaft_axis * model.probe_origin_offset_mm)

    final_branch_projection = _candidate_branch_projection(
        mesh_contact,
        main_graph=main_graph,
        network_graph=network_graph,
        branch_hint=branch_hint,
    )
    branch_hint_spec = _parse_branch_hint(branch_hint)
    branch_hint_match = None if branch_hint_spec is None or final_branch_projection is None else branch_hint_spec.matches(final_branch_projection)
    branch_hint_applied = bool(branch_hint_spec is not None and branch_hint_match)
    if branch_hint_spec is not None and not branch_hint_applied:
        refinement_warnings.append(
            f"Branch hint {branch_hint_spec.raw_value!r} could not be matched during contact refinement; the closest branch candidate was retained."
        )

    refinement = ContactRefinement(
        original_contact_world=[float(value) for value in original_contact.tolist()],
        voxel_refined_contact_world=[float(value) for value in voxel_contact.tolist()],
        refined_contact_world=[float(value) for value in mesh_contact.tolist()],
        original_contact_to_airway_distance_mm=float(original_surface_distance),
        voxel_refined_contact_to_airway_distance_mm=float(voxel_distance),
        refined_contact_to_airway_distance_mm=float(mesh_distance if np.isfinite(mesh_distance) else voxel_distance),
        voxel_to_mesh_contact_distance_mm=float(np.linalg.norm(mesh_contact - voxel_contact)),
        refinement_applied=(
            float(np.linalg.norm(voxel_contact - original_contact)) > 1e-3
            or float(np.linalg.norm(mesh_contact - original_contact)) > 1e-3
        ),
        refinement_method=mesh_method,
        voxel_refinement_method=voxel_method,
        mesh_refinement_method=mesh_method,
        candidate_hu=candidate_hu,
        candidate_branch_graph_name=(mesh_branch_graph_name if mesh_branch_graph_name is not None else voxel_branch_graph_name),
        candidate_branch_line_index=(mesh_branch if mesh_branch is not None else voxel_branch),
        branch_hint=branch_hint,
        branch_hint_applied=branch_hint_applied,
        branch_hint_match=branch_hint_match,
        warnings=refinement_warnings,
    )

    return DevicePose(
        device_model=model,
        tip_start_world=[float(value) for value in tip_start_world.tolist()],
        probe_origin_world=[float(value) for value in mesh_contact.tolist()],
        shaft_axis_world=[float(value) for value in shaft_axis.tolist()],
        video_axis_world=[float(value) for value in video_axis.tolist()],
        probe_axis_world=[float(value) for value in probe_axis.tolist()],
        lateral_axis_world=[float(value) for value in lateral_axis.tolist()],
        wall_normal_world=[float(value) for value in mesh_wall_normal.tolist()],
        voxel_wall_normal_world=[float(value) for value in voxel_wall_normal.tolist()],
        voxel_probe_axis_world=[float(value) for value in voxel_probe_axis.tolist()],
        target_world=[float(value) for value in target_world.tolist()],
        contact_refinement=refinement,
    )
