from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from ebus_simulator.centerline import CenterlineGraph, CenterlineProjection
from ebus_simulator.geometry import distance_to_mask_surface_mm
from ebus_simulator.io.mrkjson import load_first_defined_control_point
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.manifest import load_case_manifest
from ebus_simulator.models import CenterlineQuery, OrthogonalityCheck, PoseReport, PresetPose


CONTACT_AIRWAY_DISTANCE_WARN_MM = 2.0
CENTERLINE_PROJECTION_WARN_MM = 6.0
TARGET_PROJECTION_ABS_WARN_MM = 1.0
TARGET_PROJECTION_RATIO_WARN = 0.10
ORTHOGONALITY_TOLERANCE = 1e-6
EPSILON = 1e-9


def _status_from_counts(error_count: int, warning_count: int) -> str:
    if error_count:
        return "failed"
    if warning_count:
        return "warning"
    return "passed"


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return None
    return vector / norm


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return vector - (np.dot(vector, axis) * axis)


def _choose_fallback_depth_axis(shaft_axis: np.ndarray) -> np.ndarray:
    candidates = [
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    ]
    ordered = sorted(candidates, key=lambda candidate: abs(float(np.dot(candidate, shaft_axis))))
    for candidate in ordered:
        projected = _project_perpendicular(candidate, shaft_axis)
        normalized = _normalize(projected)
        if normalized is not None:
            return normalized
    raise ValueError("Failed to construct a fallback depth axis.")


def _rotate_around_axis(vector: np.ndarray, axis: np.ndarray, roll_deg: float) -> np.ndarray:
    if abs(roll_deg) <= EPSILON:
        return vector.copy()

    angle = math.radians(roll_deg)
    axis_unit = _normalize(axis)
    if axis_unit is None:
        raise ValueError("Rotation axis is undefined.")
    return (
        vector * math.cos(angle)
        + np.cross(axis_unit, vector) * math.sin(angle)
        + axis_unit * np.dot(axis_unit, vector) * (1.0 - math.cos(angle))
    )


def _orthogonality_check(shaft_axis: np.ndarray, depth_axis: np.ndarray, lateral_axis: np.ndarray) -> OrthogonalityCheck:
    shaft_norm = float(np.linalg.norm(shaft_axis))
    depth_norm = float(np.linalg.norm(depth_axis))
    lateral_norm = float(np.linalg.norm(lateral_axis))
    shaft_depth_dot = float(np.dot(shaft_axis, depth_axis))
    shaft_lateral_dot = float(np.dot(shaft_axis, lateral_axis))
    depth_lateral_dot = float(np.dot(depth_axis, lateral_axis))
    max_abs_dot = max(abs(shaft_depth_dot), abs(shaft_lateral_dot), abs(depth_lateral_dot))
    max_norm_error = max(abs(shaft_norm - 1.0), abs(depth_norm - 1.0), abs(lateral_norm - 1.0))
    return OrthogonalityCheck(
        shaft_axis_norm=shaft_norm,
        depth_axis_norm=depth_norm,
        lateral_axis_norm=lateral_norm,
        shaft_depth_dot=shaft_depth_dot,
        shaft_lateral_dot=shaft_lateral_dot,
        depth_lateral_dot=depth_lateral_dot,
        max_abs_dot=max_abs_dot,
        max_norm_error=max_norm_error,
        within_tolerance=(max_abs_dot <= ORTHOGONALITY_TOLERANCE and max_norm_error <= ORTHOGONALITY_TOLERANCE),
    )


def _query_to_model(query: CenterlineProjection | None) -> CenterlineQuery | None:
    if query is None:
        return None
    return CenterlineQuery(
        graph_name=query.graph_name,
        distance_mm=float(query.distance_mm),
        closest_point_lps=[float(value) for value in query.closest_point_lps.tolist()],
        tangent_lps=(None if query.tangent_lps is None else [float(value) for value in query.tangent_lps.tolist()]),
        tangent_defined=query.tangent_lps is not None,
        line_index=int(query.line_index),
        segment_index=int(query.segment_index),
        line_arclength_mm=float(query.line_arclength_mm),
    )


def generate_pose_report(manifest_path: str | Path, *, roll_deg: float | None = None) -> PoseReport:
    manifest = load_case_manifest(manifest_path)
    resolved_roll_deg = float(manifest.render_defaults.get("roll_deg", 0.0) if roll_deg is None else roll_deg)

    airway_lumen = load_nifti(manifest.airway_lumen_mask, kind="mask", load_data=True)
    airway_solid = load_nifti(manifest.airway_solid_mask, kind="mask", load_data=True)
    main_graph = CenterlineGraph.from_vtp(str(manifest.centerline_main), name="main")
    network_graph = CenterlineGraph.from_vtp(str(manifest.centerline_network), name="network")

    poses: list[PresetPose] = []

    for preset in manifest.presets:
        target_control_point = load_first_defined_control_point(preset.target)
        target_world = target_control_point.position_lps.astype(np.float64)

        for approach, contact_path in preset.contacts.items():
            warnings: list[str] = []
            errors: list[str] = []

            contact_control_point = load_first_defined_control_point(contact_path)
            contact_world = contact_control_point.position_lps.astype(np.float64)
            target_vector = target_world - contact_world
            contact_to_target_distance_mm = float(np.linalg.norm(target_vector))

            contact_to_airway_distance = min(
                distance_to_mask_surface_mm(contact_world, airway_lumen),
                distance_to_mask_surface_mm(contact_world, airway_solid),
            )
            if contact_to_airway_distance > CONTACT_AIRWAY_DISTANCE_WARN_MM:
                warnings.append(
                    f"Contact is {contact_to_airway_distance:.2f} mm from the airway surface."
                )

            centerline_projection = main_graph.nearest_point(contact_world)
            network_projection = network_graph.nearest_point(contact_world)
            selected_projection = centerline_projection or network_projection

            if centerline_projection is not None and centerline_projection.distance_mm > CENTERLINE_PROJECTION_WARN_MM:
                warnings.append(
                    f"Contact projects {centerline_projection.distance_mm:.2f} mm away from the main centerline."
                )
            if centerline_projection is None and network_projection is not None:
                warnings.append("Main centerline projection was unavailable; network graph tangent fallback was used.")

            if selected_projection is None or selected_projection.tangent_lps is None:
                errors.append("No valid centerline tangent could be estimated for the contact.")
                poses.append(
                    PresetPose(
                        preset_id=preset.id,
                        station=preset.station,
                        node=preset.node,
                        contact_approach=approach,
                        status="failed",
                        contact_markup_path=str(contact_path),
                        target_markup_path=str(preset.target),
                        contact_world=[float(value) for value in contact_world.tolist()],
                        target_world=[float(value) for value in target_world.tolist()],
                        contact_to_target_distance_mm=contact_to_target_distance_mm,
                        nearest_centerline_point=(None if centerline_projection is None else [float(value) for value in centerline_projection.closest_point_lps.tolist()]),
                        nearest_network_point=(None if network_projection is None else [float(value) for value in network_projection.closest_point_lps.tolist()]),
                        shaft_axis=None,
                        depth_axis=None,
                        lateral_axis=None,
                        default_depth_axis=None,
                        default_lateral_axis=None,
                        orthogonality=None,
                        target_in_default_forward_hemisphere=None,
                        target_forward_dot=None,
                        contact_to_airway_distance_mm=float(contact_to_airway_distance),
                        centerline_projection_distance_mm=(None if centerline_projection is None else float(centerline_projection.distance_mm)),
                        network_projection_distance_mm=(None if network_projection is None else float(network_projection.distance_mm)),
                        centerline_query=_query_to_model(centerline_projection),
                        network_query=_query_to_model(network_projection),
                        roll_deg=resolved_roll_deg,
                        warnings=warnings,
                        errors=errors,
                    )
                )
                continue

            shaft_axis = selected_projection.tangent_lps.astype(np.float64)
            shaft_axis = _normalize(shaft_axis)
            if shaft_axis is None:
                errors.append("Shaft axis could not be normalized.")
                poses.append(
                    PresetPose(
                        preset_id=preset.id,
                        station=preset.station,
                        node=preset.node,
                        contact_approach=approach,
                        status="failed",
                        contact_markup_path=str(contact_path),
                        target_markup_path=str(preset.target),
                        contact_world=[float(value) for value in contact_world.tolist()],
                        target_world=[float(value) for value in target_world.tolist()],
                        contact_to_target_distance_mm=contact_to_target_distance_mm,
                        nearest_centerline_point=(None if centerline_projection is None else [float(value) for value in centerline_projection.closest_point_lps.tolist()]),
                        nearest_network_point=(None if network_projection is None else [float(value) for value in network_projection.closest_point_lps.tolist()]),
                        shaft_axis=None,
                        depth_axis=None,
                        lateral_axis=None,
                        default_depth_axis=None,
                        default_lateral_axis=None,
                        orthogonality=None,
                        target_in_default_forward_hemisphere=None,
                        target_forward_dot=None,
                        contact_to_airway_distance_mm=float(contact_to_airway_distance),
                        centerline_projection_distance_mm=(None if centerline_projection is None else float(centerline_projection.distance_mm)),
                        network_projection_distance_mm=(None if network_projection is None else float(network_projection.distance_mm)),
                        centerline_query=_query_to_model(centerline_projection),
                        network_query=_query_to_model(network_projection),
                        roll_deg=resolved_roll_deg,
                        warnings=warnings,
                        errors=errors,
                    )
                )
                continue

            projected_target = _project_perpendicular(target_vector, shaft_axis)
            projected_target_norm = float(np.linalg.norm(projected_target))
            condition_ratio = 0.0 if contact_to_target_distance_mm <= EPSILON else projected_target_norm / contact_to_target_distance_mm
            poorly_conditioned = (
                projected_target_norm < TARGET_PROJECTION_ABS_WARN_MM
                or condition_ratio < TARGET_PROJECTION_RATIO_WARN
            )
            if poorly_conditioned:
                warnings.append(
                    "Projected target direction is poorly conditioned relative to the shaft axis."
                )

            raw_depth_axis = _normalize(projected_target)
            if raw_depth_axis is None:
                raw_depth_axis = _choose_fallback_depth_axis(shaft_axis)
                warnings.append("Projected target direction was degenerate; a deterministic fallback depth axis was used.")

            default_lateral_axis = _normalize(np.cross(shaft_axis, raw_depth_axis))
            if default_lateral_axis is None:
                errors.append("Lateral axis could not be constructed from shaft and raw depth axes.")
                poses.append(
                    PresetPose(
                        preset_id=preset.id,
                        station=preset.station,
                        node=preset.node,
                        contact_approach=approach,
                        status="failed",
                        contact_markup_path=str(contact_path),
                        target_markup_path=str(preset.target),
                        contact_world=[float(value) for value in contact_world.tolist()],
                        target_world=[float(value) for value in target_world.tolist()],
                        contact_to_target_distance_mm=contact_to_target_distance_mm,
                        nearest_centerline_point=(None if centerline_projection is None else [float(value) for value in centerline_projection.closest_point_lps.tolist()]),
                        nearest_network_point=(None if network_projection is None else [float(value) for value in network_projection.closest_point_lps.tolist()]),
                        shaft_axis=[float(value) for value in shaft_axis.tolist()],
                        depth_axis=None,
                        lateral_axis=None,
                        default_depth_axis=None,
                        default_lateral_axis=None,
                        orthogonality=None,
                        target_in_default_forward_hemisphere=None,
                        target_forward_dot=None,
                        contact_to_airway_distance_mm=float(contact_to_airway_distance),
                        centerline_projection_distance_mm=(None if centerline_projection is None else float(centerline_projection.distance_mm)),
                        network_projection_distance_mm=(None if network_projection is None else float(network_projection.distance_mm)),
                        centerline_query=_query_to_model(centerline_projection),
                        network_query=_query_to_model(network_projection),
                        roll_deg=resolved_roll_deg,
                        warnings=warnings,
                        errors=errors,
                    )
                )
                continue

            default_depth_axis = _normalize(np.cross(default_lateral_axis, shaft_axis))
            if default_depth_axis is None:
                errors.append("Depth axis could not be constructed from lateral and shaft axes.")
                poses.append(
                    PresetPose(
                        preset_id=preset.id,
                        station=preset.station,
                        node=preset.node,
                        contact_approach=approach,
                        status="failed",
                        contact_markup_path=str(contact_path),
                        target_markup_path=str(preset.target),
                        contact_world=[float(value) for value in contact_world.tolist()],
                        target_world=[float(value) for value in target_world.tolist()],
                        contact_to_target_distance_mm=contact_to_target_distance_mm,
                        nearest_centerline_point=(None if centerline_projection is None else [float(value) for value in centerline_projection.closest_point_lps.tolist()]),
                        nearest_network_point=(None if network_projection is None else [float(value) for value in network_projection.closest_point_lps.tolist()]),
                        shaft_axis=[float(value) for value in shaft_axis.tolist()],
                        depth_axis=None,
                        lateral_axis=[float(value) for value in default_lateral_axis.tolist()],
                        default_depth_axis=None,
                        default_lateral_axis=[float(value) for value in default_lateral_axis.tolist()],
                        orthogonality=None,
                        target_in_default_forward_hemisphere=None,
                        target_forward_dot=None,
                        contact_to_airway_distance_mm=float(contact_to_airway_distance),
                        centerline_projection_distance_mm=(None if centerline_projection is None else float(centerline_projection.distance_mm)),
                        network_projection_distance_mm=(None if network_projection is None else float(network_projection.distance_mm)),
                        centerline_query=_query_to_model(centerline_projection),
                        network_query=_query_to_model(network_projection),
                        roll_deg=resolved_roll_deg,
                        warnings=warnings,
                        errors=errors,
                    )
                )
                continue

            rolled_depth_axis = _normalize(_rotate_around_axis(default_depth_axis, shaft_axis, resolved_roll_deg))
            rolled_lateral_axis = _normalize(_rotate_around_axis(default_lateral_axis, shaft_axis, resolved_roll_deg))
            if rolled_depth_axis is None or rolled_lateral_axis is None:
                errors.append("Roll application produced an invalid pose axis.")
                poses.append(
                    PresetPose(
                        preset_id=preset.id,
                        station=preset.station,
                        node=preset.node,
                        contact_approach=approach,
                        status="failed",
                        contact_markup_path=str(contact_path),
                        target_markup_path=str(preset.target),
                        contact_world=[float(value) for value in contact_world.tolist()],
                        target_world=[float(value) for value in target_world.tolist()],
                        contact_to_target_distance_mm=contact_to_target_distance_mm,
                        nearest_centerline_point=(None if centerline_projection is None else [float(value) for value in centerline_projection.closest_point_lps.tolist()]),
                        nearest_network_point=(None if network_projection is None else [float(value) for value in network_projection.closest_point_lps.tolist()]),
                        shaft_axis=[float(value) for value in shaft_axis.tolist()],
                        depth_axis=None,
                        lateral_axis=None,
                        default_depth_axis=[float(value) for value in default_depth_axis.tolist()],
                        default_lateral_axis=[float(value) for value in default_lateral_axis.tolist()],
                        orthogonality=None,
                        target_in_default_forward_hemisphere=None,
                        target_forward_dot=None,
                        contact_to_airway_distance_mm=float(contact_to_airway_distance),
                        centerline_projection_distance_mm=(None if centerline_projection is None else float(centerline_projection.distance_mm)),
                        network_projection_distance_mm=(None if network_projection is None else float(network_projection.distance_mm)),
                        centerline_query=_query_to_model(centerline_projection),
                        network_query=_query_to_model(network_projection),
                        roll_deg=resolved_roll_deg,
                        warnings=warnings,
                        errors=errors,
                    )
                )
                continue

            orthogonality = _orthogonality_check(shaft_axis, rolled_depth_axis, rolled_lateral_axis)
            if not orthogonality.within_tolerance:
                warnings.append("Pose axes are outside the configured orthogonality tolerance.")

            target_unit = _normalize(target_vector)
            target_forward_dot = (
                None if target_unit is None else float(np.dot(target_unit, default_depth_axis))
            )
            target_in_forward_hemisphere = (
                None if target_forward_dot is None else bool(target_forward_dot >= 0.0)
            )

            status = _status_from_counts(error_count=len(errors), warning_count=len(warnings))
            poses.append(
                PresetPose(
                    preset_id=preset.id,
                    station=preset.station,
                    node=preset.node,
                    contact_approach=approach,
                    status=status,
                    contact_markup_path=str(contact_path),
                    target_markup_path=str(preset.target),
                    contact_world=[float(value) for value in contact_world.tolist()],
                    target_world=[float(value) for value in target_world.tolist()],
                    contact_to_target_distance_mm=contact_to_target_distance_mm,
                    nearest_centerline_point=(None if centerline_projection is None else [float(value) for value in centerline_projection.closest_point_lps.tolist()]),
                    nearest_network_point=(None if network_projection is None else [float(value) for value in network_projection.closest_point_lps.tolist()]),
                    shaft_axis=[float(value) for value in shaft_axis.tolist()],
                    depth_axis=[float(value) for value in rolled_depth_axis.tolist()],
                    lateral_axis=[float(value) for value in rolled_lateral_axis.tolist()],
                    default_depth_axis=[float(value) for value in default_depth_axis.tolist()],
                    default_lateral_axis=[float(value) for value in default_lateral_axis.tolist()],
                    orthogonality=orthogonality,
                    target_in_default_forward_hemisphere=target_in_forward_hemisphere,
                    target_forward_dot=target_forward_dot,
                    contact_to_airway_distance_mm=float(contact_to_airway_distance),
                    centerline_projection_distance_mm=(None if centerline_projection is None else float(centerline_projection.distance_mm)),
                    network_projection_distance_mm=(None if network_projection is None else float(network_projection.distance_mm)),
                    centerline_query=_query_to_model(centerline_projection),
                    network_query=_query_to_model(network_projection),
                    roll_deg=resolved_roll_deg,
                    warnings=warnings,
                    errors=errors,
                )
            )

    error_count = sum(len(pose.errors) for pose in poses)
    warning_count = sum(len(pose.warnings) for pose in poses)
    return PoseReport(
        manifest_path=str(manifest.manifest_path),
        case_id=manifest.case_id,
        dataset_root=str(manifest.root),
        internal_world_frame="LPS",
        roll_deg=resolved_roll_deg,
        preset_count=len(manifest.presets),
        approach_count=len(poses),
        status=_status_from_counts(error_count=error_count, warning_count=warning_count),
        centerlines={
            "main_path": str(manifest.centerline_main),
            "network_path": str(manifest.centerline_network),
            "main_line_count": main_graph.line_count,
            "main_point_count": main_graph.point_count,
            "main_segment_count": main_graph.segment_count,
            "network_line_count": network_graph.line_count,
            "network_point_count": network_graph.point_count,
            "network_segment_count": network_graph.segment_count,
        },
        poses=poses,
    )


def pose_report_to_dict(report: PoseReport) -> dict:
    return asdict(report)
