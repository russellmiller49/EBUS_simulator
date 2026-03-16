from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import ndimage

from ebus_simulator.centerline import CenterlineGraph, CenterlineProjection
from ebus_simulator.geometry import distance_to_mask_surface_mm, point_to_voxel
from ebus_simulator.models import VolumeData


EPSILON = 1e-9
CONTACT_HU_THRESHOLD = -600.0
CONTACT_SCAN_FORWARD_MM = 8.0
CONTACT_SCAN_BACKWARD_MM = 6.0
CONTACT_SCAN_SHAFT_MM = 2.0
CONTACT_SCAN_STEP_MM = 0.5
CONTACT_AMBIGUITY_SCORE_DELTA = 0.35
WALL_NORMAL_SAMPLE_DELTA_MM = 0.6


_SIGNED_DISTANCE_CACHE: dict[tuple[str, tuple[int, int, int], tuple[float, float, float]], np.ndarray] = {}


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
    refined_contact_world: list[float]
    original_contact_to_airway_distance_mm: float
    refined_contact_to_airway_distance_mm: float
    refinement_applied: bool
    refinement_method: str
    candidate_hu: float | None
    candidate_branch_line_index: int | None
    warnings: list[str] = field(default_factory=list)


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
    return vector / norm


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return vector - (float(np.dot(vector, axis)) * axis)


def _sample_scalar(volume: VolumeData, point_lps: np.ndarray, *, order: int, cval: float) -> float:
    if volume.data is None:
        raise ValueError(f"Volume {volume.path} must be loaded for device sampling.")
    voxel = point_to_voxel(point_lps, volume)
    value = ndimage.map_coordinates(
        np.asarray(volume.data, dtype=np.float32),
        [[float(voxel[0])], [float(voxel[1])], [float(voxel[2])]],
        order=order,
        mode="constant",
        cval=cval,
    )
    return float(value[0])


def _signed_distance_volume(mask_volume: VolumeData) -> np.ndarray:
    if mask_volume.data is None:
        raise ValueError(f"Mask volume {mask_volume.path} must be loaded for wall-normal estimation.")
    mask = np.asarray(mask_volume.data > 0, dtype=bool)
    cache_key = (
        str(mask_volume.path),
        tuple(int(value) for value in mask.shape[:3]),
        tuple(float(value) for value in mask_volume.voxel_sizes_mm[:3]),
    )
    cached = _SIGNED_DISTANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    outside = ndimage.distance_transform_edt(~mask, sampling=mask_volume.voxel_sizes_mm[:3])
    inside = ndimage.distance_transform_edt(mask, sampling=mask_volume.voxel_sizes_mm[:3])
    signed_distance = outside - inside
    _SIGNED_DISTANCE_CACHE[cache_key] = signed_distance.astype(np.float32)
    return _SIGNED_DISTANCE_CACHE[cache_key]


def _sample_signed_distance(mask_volume: VolumeData, point_lps: np.ndarray) -> float:
    return _sample_scalar(
        VolumeData(
            path=mask_volume.path,
            kind=mask_volume.kind,
            shape=mask_volume.shape,
            dtype="float32",
            affine_ras=mask_volume.affine_ras,
            affine_lps=mask_volume.affine_lps,
            inverse_affine_lps=mask_volume.inverse_affine_lps,
            voxel_sizes_mm=mask_volume.voxel_sizes_mm,
            axis_codes_ras=mask_volume.axis_codes_ras,
            data=_signed_distance_volume(mask_volume),
        ),
        point_lps,
        order=1,
        cval=float(np.max(_signed_distance_volume(mask_volume))),
    )


def _estimate_wall_normal(point_lps: np.ndarray, airway_solid: VolumeData) -> np.ndarray | None:
    gradient = np.zeros(3, dtype=np.float64)
    for axis_index in range(3):
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = 1.0
        forward = _sample_signed_distance(airway_solid, point_lps + (axis * WALL_NORMAL_SAMPLE_DELTA_MM))
        backward = _sample_signed_distance(airway_solid, point_lps - (axis * WALL_NORMAL_SAMPLE_DELTA_MM))
        gradient[axis_index] = (forward - backward) / (2.0 * WALL_NORMAL_SAMPLE_DELTA_MM)
    normalized = _normalize(gradient)
    if normalized is None:
        return None
    return normalized


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


def _fallback_wall_normal(contact_world: np.ndarray, shaft_axis: np.ndarray, projection: CenterlineProjection | None, fallback_probe_axis: np.ndarray) -> np.ndarray:
    if projection is not None:
        radial = _project_perpendicular(contact_world - projection.closest_point_lps, shaft_axis)
        radial_normalized = _normalize(radial)
        if radial_normalized is not None:
            return radial_normalized
    return fallback_probe_axis


def _candidate_branch_projection(
    point_lps: np.ndarray,
    *,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
) -> CenterlineProjection | None:
    projection = main_graph.nearest_point(point_lps)
    if projection is not None:
        return projection
    if network_graph is None:
        return None
    return network_graph.nearest_point(point_lps)


def refine_airway_contact(
    seed_contact_world: np.ndarray,
    *,
    shaft_axis: np.ndarray,
    probe_axis: np.ndarray,
    ct_volume: VolumeData,
    airway_lumen: VolumeData,
    airway_solid: VolumeData,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None = None,
) -> ContactRefinement:
    original_surface_distance = _surface_distance(seed_contact_world, airway_lumen, airway_solid)
    warnings: list[str] = []

    seed_projection = _candidate_branch_projection(
        seed_contact_world,
        main_graph=main_graph,
        network_graph=network_graph,
    )
    seed_branch = None if seed_projection is None else int(seed_projection.line_index)

    offsets_us = np.arange(-CONTACT_SCAN_BACKWARD_MM, CONTACT_SCAN_FORWARD_MM + CONTACT_SCAN_STEP_MM, CONTACT_SCAN_STEP_MM, dtype=np.float64)
    offsets_b = np.arange(-CONTACT_SCAN_SHAFT_MM, CONTACT_SCAN_SHAFT_MM + CONTACT_SCAN_STEP_MM, CONTACT_SCAN_STEP_MM, dtype=np.float64)
    candidates: list[tuple[float, np.ndarray, float, int | None]] = []

    for offset_b in offsets_b:
        for offset_us in offsets_us:
            point_lps = seed_contact_world + (offset_us * probe_axis) + (offset_b * shaft_axis)
            hu_value = _sample_scalar(ct_volume, point_lps, order=1, cval=-1000.0)
            if hu_value <= CONTACT_HU_THRESHOLD:
                continue

            projection = _candidate_branch_projection(
                point_lps,
                main_graph=main_graph,
                network_graph=network_graph,
            )
            surface_distance = _surface_distance(point_lps, airway_lumen, airway_solid)
            branch_penalty = 0.0
            branch_index = None
            direction_penalty = 0.0

            if projection is not None:
                branch_index = int(projection.line_index)
                if seed_branch is not None and branch_index != seed_branch:
                    branch_penalty = 1.5
                radial_vector = point_lps - projection.closest_point_lps
                radial_unit = _normalize(_project_perpendicular(radial_vector, projection.tangent_lps if projection.tangent_lps is not None else shaft_axis))
                if radial_unit is not None:
                    direction_penalty = max(0.0, 0.75 - float(np.dot(radial_unit, probe_axis)))

            score = (
                surface_distance
                + (0.10 * abs(float(offset_us)))
                + (0.15 * abs(float(offset_b)))
                + branch_penalty
                + direction_penalty
            )
            candidates.append((score, point_lps, hu_value, branch_index))
            break

    if not candidates:
        warnings.append("Contact refinement found no tissue candidate above the HU threshold; the markup contact was retained.")
        return ContactRefinement(
            original_contact_world=[float(value) for value in seed_contact_world.tolist()],
            refined_contact_world=[float(value) for value in seed_contact_world.tolist()],
            original_contact_to_airway_distance_mm=float(original_surface_distance),
            refined_contact_to_airway_distance_mm=float(original_surface_distance),
            refinement_applied=False,
            refinement_method="seed_fallback",
            candidate_hu=None,
            candidate_branch_line_index=seed_branch,
            warnings=warnings,
        )

    candidates.sort(key=lambda item: item[0])
    best_score, best_point, best_hu, best_branch = candidates[0]
    refined_surface_distance = _surface_distance(best_point, airway_lumen, airway_solid)

    if len(candidates) > 1:
        second_score, second_point, _, _ = candidates[1]
        separation = float(np.linalg.norm(second_point - best_point))
        if abs(second_score - best_score) <= CONTACT_AMBIGUITY_SCORE_DELTA and separation > 1.0:
            warnings.append("Contact refinement was ambiguous; the best airway-wall candidate was chosen deterministically.")

    if refined_surface_distance > (original_surface_distance + 0.75):
        warnings.append("Contact refinement did not improve airway-wall proximity; the markup contact was retained.")
        return ContactRefinement(
            original_contact_world=[float(value) for value in seed_contact_world.tolist()],
            refined_contact_world=[float(value) for value in seed_contact_world.tolist()],
            original_contact_to_airway_distance_mm=float(original_surface_distance),
            refined_contact_to_airway_distance_mm=float(original_surface_distance),
            refinement_applied=False,
            refinement_method="seed_retained",
            candidate_hu=best_hu,
            candidate_branch_line_index=best_branch,
            warnings=warnings,
        )

    if refined_surface_distance > 2.0:
        warnings.append(f"Refined contact remains {refined_surface_distance:.2f} mm from the airway wall.")

    return ContactRefinement(
        original_contact_world=[float(value) for value in seed_contact_world.tolist()],
        refined_contact_world=[float(value) for value in best_point.tolist()],
        original_contact_to_airway_distance_mm=float(original_surface_distance),
        refined_contact_to_airway_distance_mm=float(refined_surface_distance),
        refinement_applied=(float(np.linalg.norm(best_point - seed_contact_world)) > 1e-3),
        refinement_method="ct_threshold_wall_search",
        candidate_hu=float(best_hu),
        candidate_branch_line_index=best_branch,
        warnings=warnings,
    )


def build_device_pose(
    pose,
    *,
    device_name: str,
    ct_volume: VolumeData,
    airway_lumen: VolumeData,
    airway_solid: VolumeData,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None = None,
    refine_contact: bool = True,
) -> DevicePose:
    model = get_cp_ebus_device_model(device_name)

    original_contact = np.asarray(pose.contact_world, dtype=np.float64)
    target_world = np.asarray(pose.target_world, dtype=np.float64)
    shaft_axis = _normalize(np.asarray(pose.shaft_axis, dtype=np.float64))
    if shaft_axis is None:
        raise ValueError(f"Preset {pose.preset_id!r} approach {pose.contact_approach!r} does not have a valid shaft axis.")

    fallback_probe_axis = _normalize(np.asarray(pose.depth_axis if pose.depth_axis is not None else pose.default_depth_axis, dtype=np.float64))
    if fallback_probe_axis is None:
        raise ValueError(f"Preset {pose.preset_id!r} approach {pose.contact_approach!r} does not have a valid depth/probe axis.")

    refinement = (
        refine_airway_contact(
            original_contact,
            shaft_axis=shaft_axis,
            probe_axis=fallback_probe_axis,
            ct_volume=ct_volume,
            airway_lumen=airway_lumen,
            airway_solid=airway_solid,
            main_graph=main_graph,
            network_graph=network_graph,
        )
        if refine_contact
        else ContactRefinement(
            original_contact_world=[float(value) for value in original_contact.tolist()],
            refined_contact_world=[float(value) for value in original_contact.tolist()],
            original_contact_to_airway_distance_mm=float(_surface_distance(original_contact, airway_lumen, airway_solid)),
            refined_contact_to_airway_distance_mm=float(_surface_distance(original_contact, airway_lumen, airway_solid)),
            refinement_applied=False,
            refinement_method="disabled",
            candidate_hu=None,
            candidate_branch_line_index=(None if pose.centerline_query is None else pose.centerline_query.line_index),
            warnings=[],
        )
    )

    refined_contact = np.asarray(refinement.refined_contact_world, dtype=np.float64)
    projection = _candidate_branch_projection(
        refined_contact,
        main_graph=main_graph,
        network_graph=network_graph,
    )
    wall_normal = _estimate_wall_normal(refined_contact, airway_solid)
    if wall_normal is None:
        wall_normal = _fallback_wall_normal(refined_contact, shaft_axis, projection, fallback_probe_axis)
        refinement.warnings.append("Wall normal estimation was degenerate; a centerline-radial fallback was used.")

    projected_wall_normal = _project_perpendicular(wall_normal, shaft_axis)
    probe_axis = _normalize(projected_wall_normal)
    if probe_axis is None:
        probe_axis = fallback_probe_axis
        refinement.warnings.append("Wall-normal projection was degenerate; the fallback probe axis was used.")

    target_vector = target_world - refined_contact
    if float(np.dot(target_vector, probe_axis)) < 0.0:
        probe_axis = -probe_axis
        if float(np.dot(projected_wall_normal, probe_axis)) < 0.0:
            wall_normal = -wall_normal

    lateral_axis = _normalize(np.cross(shaft_axis, probe_axis))
    if lateral_axis is None:
        raise ValueError("Failed to construct the lateral axis for the CP-EBUS device pose.")

    video_axis = _rotation_toward(shaft_axis, probe_axis, model.video_axis_offset_deg)
    tip_start_world = refined_contact - (shaft_axis * model.probe_origin_offset_mm)

    return DevicePose(
        device_model=model,
        tip_start_world=[float(value) for value in tip_start_world.tolist()],
        probe_origin_world=[float(value) for value in refined_contact.tolist()],
        shaft_axis_world=[float(value) for value in shaft_axis.tolist()],
        video_axis_world=[float(value) for value in video_axis.tolist()],
        probe_axis_world=[float(value) for value in probe_axis.tolist()],
        lateral_axis_world=[float(value) for value in lateral_axis.tolist()],
        wall_normal_world=[float(value) for value in wall_normal.tolist()],
        target_world=[float(value) for value in target_world.tolist()],
        contact_refinement=refinement,
    )
