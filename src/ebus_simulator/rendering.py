from __future__ import annotations

from csv import DictWriter
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ebus_simulator.annotations import (
    _add_panel_label,
    _annotate_legend_and_labels,
    _apply_contour_overlay,
    _color_tuple,
    _draw_cross_marker,
    _filter_mask_components,
)
from ebus_simulator.cutaway import CutawayDisplay, build_display_cutaway
from ebus_simulator.device import DevicePose, build_device_pose, _parse_branch_hint
from ebus_simulator.render_engines import RenderEngine, RenderRequest, RenderResult, parse_render_engine
from ebus_simulator.render_profiles import load_consistency_profile
from ebus_simulator.render_state import (
    DEFAULT_RENDER_MODE,
    _LOCAL_POSE_OPTIMIZATION_CACHE,
    BatchRenderIndex,
    CutawayConfig,
    LocalPoseOptimizationResult,
    OverlayConfig,
    OverlayLayer,
    RenderContext,
    RenderIndexEntry,
    RenderMetadata,
    RenderedPreset,
    SourceSection,
    _get_mask_volume,
    _overlay_summary,
    _resolve_branch_shift_seed,
    _resolve_cutaway_config,
    _resolve_overlay_config,
    _resolve_pose,
    _resolve_preset_manifest,
    _resolve_slice_thickness_mm,
    build_render_context,
)
from ebus_simulator.transforms import (
    DEFAULT_SLAB_SAMPLES,
    OPTIMIZATION_EPSILON,
    _angle_deg,
    _build_sector_grid,
    _fan_target_row_col,
    _map_plane_to_fan,
    _normalize,
    _plane_point_to_pixel,
    _points_to_voxel,
    _preprocess_source_section,
    _project_perpendicular,
    _sample_contact_plane,
    _sample_plane,
    _sample_slab,
    _window_ct,
)


CONTEXT_AXIS_LENGTH_MM = 12.0
CONTEXT_CENTERLINE_WINDOW_MM = 28.0
DEFAULT_DEVICE_NAME = "bf_uc180f"
DEFAULT_REFERENCE_FOV_MM = 100.0
DEFAULT_SOURCE_OBLIQUE_SIZE_MM = 51.79
FLAGGED_BRANCH_SHIFT_MM = (-4.0, 0.0, 4.0)
FLAGGED_ROLL_DELTA_DEG = (0.0,)
FLAGGED_AXIS_SIGN_OVERRIDES = ()
_LOCAL_POSE_OPTIMIZATION_CACHE: dict[tuple[object, ...], LocalPoseOptimizationResult] = {}
DEFAULT_CONSISTENCY_PROFILE = load_consistency_profile().settings

AIRWAY_LUMEN_COLOR = np.asarray([0.22, 0.92, 0.94], dtype=np.float32)
AIRWAY_WALL_COLOR = np.asarray([0.16, 0.47, 0.96], dtype=np.float32)
STATION_COLOR = np.asarray([0.96, 0.84, 0.28], dtype=np.float32)
VESSEL_OVERLAY_PALETTE = [
    np.asarray([0.95, 0.45, 0.32], dtype=np.float32),
    np.asarray([0.83, 0.27, 0.27], dtype=np.float32),
    np.asarray([0.45, 0.65, 0.95], dtype=np.float32),
    np.asarray([0.61, 0.48, 0.92], dtype=np.float32),
    np.asarray([0.32, 0.75, 0.52], dtype=np.float32),
]
TARGET_MARKER_COLOR = np.asarray([1.0, 0.25, 0.25], dtype=np.float32)
CONTACT_MARKER_COLOR = np.asarray([1.0, 0.94, 0.55], dtype=np.float32)
CENTERLINE_CONTEXT_COLOR = (160, 160, 160)
FAN_BOUNDARY_COLOR = np.asarray([0.82, 0.82, 0.82], dtype=np.float32)
VIDEO_AXIS_CONTEXT_COLOR = (255, 205, 120)
PROBE_AXIS_CONTEXT_COLOR = (255, 120, 120)
FRUSTUM_CONTEXT_COLOR = (200, 200, 200)
CUTAWAY_CONTEXT_EDGE_COLOR = (255, 214, 132)
CUTAWAY_TRIANGLE_EPSILON_MM = 0.35
CONTEXT_SCREEN_X_BASIS = np.asarray([0.92, 0.35, 0.0], dtype=np.float64)
CONTEXT_SCREEN_Y_BASIS = np.asarray([0.0, 0.24, -0.82], dtype=np.float64)
CONTEXT_VIEW_DIRECTION = np.cross(CONTEXT_SCREEN_X_BASIS, CONTEXT_SCREEN_Y_BASIS)


def _compute_contact_to_centerline_mm(context: RenderContext, contact_world: np.ndarray) -> tuple[float | None, str | None]:
    projection = context.main_graph.nearest_point(contact_world)
    if projection is not None:
        return float(projection.distance_mm), "main"
    projection = context.network_graph.nearest_point(contact_world)
    if projection is not None:
        return float(projection.distance_mm), "network"
    return None, None


def _compute_station_overlap_fraction_in_fan(
    context: RenderContext,
    *,
    preset_manifest,
    contact_world: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    thickness_axis: np.ndarray,
    width: int,
    height: int,
    source_oblique_size_mm: float,
    max_depth_mm: float,
    sector_angle_deg: float,
    slice_thickness_mm: float,
) -> float:
    station_mask = _get_mask_volume(context, preset_manifest.station_mask)
    source_mask, _, _, _ = _sample_contact_plane(
        data=np.asarray(station_mask.data, dtype=np.float32),
        inverse_affine_lps=station_mask.inverse_affine_lps,
        origin_world=contact_world,
        depth_axis=probe_axis,
        lateral_axis=shaft_axis,
        thickness_axis=thickness_axis,
        depth_max_mm=source_oblique_size_mm,
        lateral_half_width_mm=(source_oblique_size_mm / 2.0),
        width=width,
        height=height,
        slice_thickness_mm=slice_thickness_mm,
        order=0,
        cval=0.0,
    )
    depth_grid, lateral_grid, sector_mask, _ = _build_sector_grid(width, height, max_depth_mm, sector_angle_deg)
    fan_mask = _map_plane_to_fan(
        source_mask,
        source_forward_max_mm=source_oblique_size_mm,
        source_shaft_half_width_mm=(source_oblique_size_mm / 2.0),
        depth_grid_mm=depth_grid,
        display_lateral_grid_mm=lateral_grid,
        sector_mask=sector_mask,
        order=0,
        cval=0.0,
    ) > 0.0
    denominator = int(np.count_nonzero(sector_mask))
    if denominator <= 0:
        return 0.0
    return float(np.count_nonzero(np.logical_and(fan_mask, sector_mask)) / denominator)


def compute_pose_review_metrics(
    context: RenderContext,
    *,
    preset_manifest,
    target_world: np.ndarray,
    contact_world: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    thickness_axis: np.ndarray | None,
    warnings: list[str],
    width: int,
    height: int,
    source_oblique_size_mm: float,
    max_depth_mm: float,
    sector_angle_deg: float,
    slice_thickness_mm: float,
    nUS_delta_deg_from_voxel_baseline: float,
    contact_delta_mm_from_voxel_baseline: float,
) -> dict[str, object]:
    resolved_thickness_axis = _normalize(thickness_axis) if thickness_axis is not None else _normalize(np.cross(shaft_axis, probe_axis))
    if resolved_thickness_axis is None:
        resolved_thickness_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    target_offset = target_world - contact_world
    target_depth_mm = float(np.dot(target_offset, probe_axis))
    target_lateral_offset_mm = float(np.dot(target_offset, shaft_axis))
    target_in_forward_hemisphere = bool(target_depth_mm > 0.0)
    _, _, _, max_lateral_mm = _build_sector_grid(width, height, max_depth_mm, sector_angle_deg)
    target_in_sector = (
        _fan_target_row_col(
            contact_world=contact_world,
            target_world=target_world,
            probe_axis=probe_axis,
            shaft_axis=shaft_axis,
            max_depth_mm=max_depth_mm,
            sector_angle_deg=sector_angle_deg,
            width=width,
            height=height,
            max_lateral_mm=max_lateral_mm,
        )
        is not None
    )

    contact_to_centerline_mm, contact_centerline_source = _compute_contact_to_centerline_mm(context, contact_world)
    station_overlap_fraction_in_fan = _compute_station_overlap_fraction_in_fan(
        context,
        preset_manifest=preset_manifest,
        contact_world=contact_world,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        thickness_axis=resolved_thickness_axis,
        width=width,
        height=height,
        source_oblique_size_mm=source_oblique_size_mm,
        max_depth_mm=max_depth_mm,
        sector_angle_deg=sector_angle_deg,
        slice_thickness_mm=slice_thickness_mm,
    )

    contact_refinement_ambiguity = any("ambiguous" in warning.lower() for warning in warnings)
    return {
        "contact_to_centerline_mm": contact_to_centerline_mm,
        "contact_centerline_source": contact_centerline_source,
        "target_depth_mm": target_depth_mm,
        "target_lateral_offset_mm": target_lateral_offset_mm,
        "target_in_sector": target_in_sector,
        "target_in_forward_hemisphere": target_in_forward_hemisphere,
        "station_overlap_fraction_in_fan": station_overlap_fraction_in_fan,
        "nUS_delta_deg_from_voxel_baseline": float(nUS_delta_deg_from_voxel_baseline),
        "contact_delta_mm_from_voxel_baseline": float(contact_delta_mm_from_voxel_baseline),
        "contact_refinement_ambiguity": contact_refinement_ambiguity,
        "warnings": list(warnings),
    }


def _mask_fraction(mask: np.ndarray, reference_mask: np.ndarray) -> float:
    denominator = int(np.count_nonzero(reference_mask))
    if denominator <= 0:
        return 0.0
    return float(np.count_nonzero(np.asarray(mask, dtype=bool) & np.asarray(reference_mask, dtype=bool)) / denominator)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    masked = np.asarray(values, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if masked.size == 0:
        return None
    return float(np.mean(masked))


def _masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float | None:
    masked = np.asarray(values, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if masked.size == 0:
        return None
    return float(np.percentile(masked, percentile))


def _build_target_region_mask(
    *,
    depth_grid_mm: np.ndarray,
    lateral_grid_mm: np.ndarray,
    sector_mask: np.ndarray,
    target_depth_mm: float | None,
    target_lateral_offset_mm: float | None,
    radius_mm: float | None = None,
) -> np.ndarray:
    if target_depth_mm is None or target_lateral_offset_mm is None:
        return np.zeros_like(sector_mask, dtype=bool)
    resolved_radius_mm = float(DEFAULT_CONSISTENCY_PROFILE.target_region_radius_mm if radius_mm is None else radius_mm)
    depth_delta = depth_grid_mm - float(target_depth_mm)
    lateral_delta = lateral_grid_mm - float(target_lateral_offset_mm)
    return np.asarray(
        np.logical_and((depth_delta ** 2 + lateral_delta ** 2) <= resolved_radius_mm ** 2, sector_mask),
        dtype=bool,
    )


def _metric_float(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_render_consistency_bucket(
    metrics: Mapping[str, object],
    *,
    profile=None,
) -> str:
    resolved_profile = DEFAULT_CONSISTENCY_PROFILE if profile is None else profile
    target_contrast = _metric_float(metrics, "target_region_contrast_vs_sector")
    target_coverage = _metric_float(metrics, "target_sector_coverage_fraction")
    near_field_wall_occupancy = _metric_float(metrics, "near_field_wall_occupancy_fraction")
    wall_contrast = _metric_float(metrics, "wall_region_contrast_vs_sector")
    empty_sector_fraction = _metric_float(metrics, "empty_sector_fraction")
    non_background_occupancy = _metric_float(metrics, "non_background_occupancy_fraction")

    if (
        target_contrast is not None
        and target_coverage is not None
        and target_contrast >= float(resolved_profile.target_prominent_min_contrast)
        and target_coverage >= float(resolved_profile.target_prominent_min_coverage)
    ):
        return "target_prominent"

    if (
        near_field_wall_occupancy is not None
        and wall_contrast is not None
        and near_field_wall_occupancy >= float(resolved_profile.wall_dominant_min_near_field_occupancy)
        and wall_contrast >= float(resolved_profile.wall_dominant_min_contrast)
        and (
            target_contrast is None
            or target_contrast <= float(resolved_profile.wall_dominant_max_target_contrast)
        )
    ):
        return "wall_dominant"

    if (
        empty_sector_fraction is not None
        and non_background_occupancy is not None
        and empty_sector_fraction >= float(resolved_profile.sparse_empty_fraction)
        and non_background_occupancy <= float(resolved_profile.sparse_non_background_max)
    ):
        return "sparse_empty_dominant"

    return "mixed_occupancy"


def compute_render_consistency_metrics(
    *,
    image_gray: np.ndarray,
    sector_mask: np.ndarray,
    depth_grid_mm: np.ndarray,
    lateral_grid_mm: np.ndarray,
    max_depth_mm: float,
    sector_angle_deg: float,
    target_depth_mm: float | None,
    target_lateral_offset_mm: float | None,
    target_region_mask: np.ndarray | None = None,
    airway_wall_mask: np.ndarray | None = None,
    vessel_mask: np.ndarray | None = None,
    normalization_method: str,
    normalization_reference_percentile: float | None = None,
    normalization_reference_value: float | None = None,
    normalization_aux_percentile: float | None = None,
    normalization_aux_value: float | None = None,
    normalization_lower_bound: float | None = None,
    normalization_upper_bound: float | None = None,
    compression_gain_factor: float | None = None,
    signal_threshold: float | None = None,
    near_field_fraction: float | None = None,
) -> dict[str, object]:
    resolved_profile = DEFAULT_CONSISTENCY_PROFILE
    resolved_signal_threshold = float(resolved_profile.signal_threshold if signal_threshold is None else signal_threshold)
    resolved_near_field_fraction = float(resolved_profile.near_field_fraction if near_field_fraction is None else near_field_fraction)
    gray = np.asarray(image_gray, dtype=np.float32)
    sector = np.asarray(sector_mask, dtype=bool)
    target_mask = (
        _build_target_region_mask(
            depth_grid_mm=depth_grid_mm,
            lateral_grid_mm=lateral_grid_mm,
            sector_mask=sector,
            target_depth_mm=target_depth_mm,
            target_lateral_offset_mm=target_lateral_offset_mm,
        )
        if target_region_mask is None
        else np.asarray(target_region_mask, dtype=bool) & sector
    )
    wall = np.zeros_like(sector, dtype=bool) if airway_wall_mask is None else np.asarray(airway_wall_mask, dtype=bool) & sector
    vessel = np.zeros_like(sector, dtype=bool) if vessel_mask is None else np.asarray(vessel_mask, dtype=bool) & sector
    anatomy = wall | vessel | target_mask
    near_field_mask = sector & (np.asarray(depth_grid_mm, dtype=np.float32) <= (float(max_depth_mm) * resolved_near_field_fraction))
    signal_mask = sector & (gray > resolved_signal_threshold)

    sector_values = gray[sector]
    sector_mean = None if sector_values.size == 0 else float(np.mean(sector_values))
    target_mean = _masked_mean(gray, target_mask)
    wall_mean = _masked_mean(gray, wall)
    vessel_mean = _masked_mean(gray, vessel)

    target_in_sector = None
    target_centerline_offset_fraction = None
    target_sector_margin_mm = None
    target_sector_margin_fraction = None
    if target_depth_mm is not None and target_lateral_offset_mm is not None:
        half_tan = float(np.tan(np.deg2rad(float(sector_angle_deg) / 2.0)))
        allowed_lateral_mm = max(0.0, float(target_depth_mm) * half_tan)
        target_in_sector = bool(
            0.0 <= float(target_depth_mm) <= float(max_depth_mm)
            and abs(float(target_lateral_offset_mm)) <= (allowed_lateral_mm + 1e-9)
        )
        target_sector_margin_mm = float(allowed_lateral_mm - abs(float(target_lateral_offset_mm)))
        if allowed_lateral_mm > 1e-9:
            target_centerline_offset_fraction = float(abs(float(target_lateral_offset_mm)) / allowed_lateral_mm)
            target_sector_margin_fraction = float(target_sector_margin_mm / allowed_lateral_mm)

    sector_p05 = None if sector_values.size == 0 else float(np.percentile(sector_values, 5.0))
    sector_p50 = None if sector_values.size == 0 else float(np.percentile(sector_values, 50.0))
    sector_p95 = None if sector_values.size == 0 else float(np.percentile(sector_values, 95.0))
    sector_p99 = None if sector_values.size == 0 else float(np.percentile(sector_values, 99.0))
    sector_std = None if sector_values.size == 0 else float(np.std(sector_values))

    metrics = {
        "normalization_method": normalization_method,
        "normalization_reference_percentile": normalization_reference_percentile,
        "normalization_reference_value": normalization_reference_value,
        "normalization_aux_percentile": normalization_aux_percentile,
        "normalization_aux_value": normalization_aux_value,
        "normalization_lower_bound": normalization_lower_bound,
        "normalization_upper_bound": normalization_upper_bound,
        "compression_gain_factor": compression_gain_factor,
        "signal_threshold": resolved_signal_threshold,
        "near_field_fraction": resolved_near_field_fraction,
        "target_depth_mm": target_depth_mm,
        "target_lateral_offset_mm": target_lateral_offset_mm,
        "target_distance_from_sector_centerline_mm": (None if target_lateral_offset_mm is None else float(abs(target_lateral_offset_mm))),
        "target_centerline_offset_fraction": target_centerline_offset_fraction,
        "target_sector_margin_mm": target_sector_margin_mm,
        "target_sector_margin_fraction": target_sector_margin_fraction,
        "target_in_sector": target_in_sector,
        "target_sector_coverage_fraction": _mask_fraction(target_mask, sector),
        "airway_wall_occupancy_fraction": _mask_fraction(wall, sector),
        "vessel_occupancy_fraction": _mask_fraction(vessel, sector),
        "anatomy_occupancy_fraction": _mask_fraction(anatomy, sector),
        "non_background_occupancy_fraction": _mask_fraction(signal_mask, sector),
        "empty_sector_fraction": 1.0 - _mask_fraction(signal_mask, sector),
        "near_field_wall_occupancy_fraction": _mask_fraction(wall, near_field_mask),
        "near_field_brightness_mean": _masked_mean(gray, near_field_mask),
        "near_field_brightness_p90": _masked_percentile(gray, near_field_mask, 90.0),
        "sector_brightness_mean": sector_mean,
        "sector_brightness_std": sector_std,
        "sector_brightness_p05": sector_p05,
        "sector_brightness_p50": sector_p50,
        "sector_brightness_p95": sector_p95,
        "sector_brightness_p99": sector_p99,
        "target_region_mean_intensity": target_mean,
        "target_region_contrast_vs_sector": (None if target_mean is None or sector_mean is None else float(target_mean - sector_mean)),
        "wall_region_mean_intensity": wall_mean,
        "wall_region_contrast_vs_sector": (None if wall_mean is None or sector_mean is None else float(wall_mean - sector_mean)),
        "vessel_region_mean_intensity": vessel_mean,
        "vessel_region_contrast_vs_sector": (None if vessel_mean is None or sector_mean is None else float(vessel_mean - sector_mean)),
    }
    metrics["consistency_bucket"] = classify_render_consistency_bucket(metrics, profile=resolved_profile)
    return metrics


def _local_pose_objective(
    *,
    metrics: dict[str, object],
    device_pose: DevicePose,
    baseline_device_pose: DevicePose,
    branch_shift_mm: float,
    roll_offset_deg: float,
    base_roll_offset_deg: float,
    axis_sign_override: str | None,
    base_axis_sign_override: str | None,
) -> tuple[float | int, ...]:
    wall_alignment = float(
        np.dot(
            np.asarray(device_pose.wall_normal_world, dtype=np.float64),
            np.asarray(device_pose.probe_axis_world, dtype=np.float64),
        )
    )
    baseline_contact = np.asarray(baseline_device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    candidate_contact = np.asarray(device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    contact_shift_mm = float(np.linalg.norm(candidate_contact - baseline_contact))
    probe_axis_delta_deg = _angle_deg(
        np.asarray(baseline_device_pose.probe_axis_world, dtype=np.float64),
        np.asarray(device_pose.probe_axis_world, dtype=np.float64),
    )
    axis_penalty = 0 if axis_sign_override in {None, base_axis_sign_override} else -1

    return (
        int(bool(metrics["target_in_sector"])),
        int(bool(metrics["target_in_forward_hemisphere"])),
        int(bool(device_pose.contact_refinement.branch_hint_applied)),
        int(float(metrics["station_overlap_fraction_in_fan"]) >= 0.003),
        round(float(metrics["station_overlap_fraction_in_fan"]), 6),
        round(wall_alignment, 6),
        int(not bool(metrics["contact_refinement_ambiguity"])),
        -round(abs(float(metrics["target_lateral_offset_mm"])), 6),
        -round(float(metrics["nUS_delta_deg_from_voxel_baseline"]), 6),
        -round(contact_shift_mm, 6),
        -round(probe_axis_delta_deg, 6),
        -round(abs(float(roll_offset_deg) - float(base_roll_offset_deg)), 6),
        axis_penalty,
        -round(abs(float(branch_shift_mm)), 6),
    )


def _optimize_flagged_pose_locally(
    context: RenderContext,
    *,
    pose,
    preset_manifest,
    device: str,
    branch_hint: str,
    base_device_pose: DevicePose,
    base_roll_offset_deg: float,
    base_axis_sign_override: str | None,
    width: int,
    height: int,
    source_oblique_size_mm: float,
    max_depth_mm: float,
    sector_angle_deg: float,
    slice_thickness_mm: float,
) -> LocalPoseOptimizationResult | None:
    branch_keys = _candidate_branch_keys_for_hint(context, branch_hint)
    if not branch_keys:
        return None

    baseline_metrics = compute_pose_review_metrics(
        context,
        preset_manifest=preset_manifest,
        target_world=np.asarray(base_device_pose.target_world, dtype=np.float64),
        contact_world=np.asarray(base_device_pose.contact_refinement.refined_contact_world, dtype=np.float64),
        probe_axis=np.asarray(base_device_pose.probe_axis_world, dtype=np.float64),
        shaft_axis=np.asarray(base_device_pose.shaft_axis_world, dtype=np.float64),
        thickness_axis=np.asarray(base_device_pose.lateral_axis_world, dtype=np.float64),
        warnings=list(device_pose_warning for device_pose_warning in base_device_pose.contact_refinement.warnings),
        width=width,
        height=height,
        source_oblique_size_mm=source_oblique_size_mm,
        max_depth_mm=max_depth_mm,
        sector_angle_deg=sector_angle_deg,
        slice_thickness_mm=slice_thickness_mm,
        nUS_delta_deg_from_voxel_baseline=_angle_deg(
            np.asarray(base_device_pose.voxel_probe_axis_world, dtype=np.float64),
            np.asarray(base_device_pose.probe_axis_world, dtype=np.float64),
        ),
        contact_delta_mm_from_voxel_baseline=float(base_device_pose.contact_refinement.voxel_to_mesh_contact_distance_mm),
    )
    best = LocalPoseOptimizationResult(
        device_pose=base_device_pose,
        branch_shift_mm=0.0,
        roll_offset_deg=float(base_roll_offset_deg),
        axis_sign_override=base_axis_sign_override,
        objective=_local_pose_objective(
            metrics=baseline_metrics,
            device_pose=base_device_pose,
            baseline_device_pose=base_device_pose,
            branch_shift_mm=0.0,
            roll_offset_deg=float(base_roll_offset_deg),
            base_roll_offset_deg=float(base_roll_offset_deg),
            axis_sign_override=base_axis_sign_override,
            base_axis_sign_override=base_axis_sign_override,
        ),
        metrics=baseline_metrics,
    )

    baseline_contact = np.asarray(base_device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    fallback_probe_axis = np.asarray(base_device_pose.probe_axis_world, dtype=np.float64)
    target_world = np.asarray(base_device_pose.target_world, dtype=np.float64)

    axis_options: list[str | None] = []
    for value in (base_axis_sign_override,) + FLAGGED_AXIS_SIGN_OVERRIDES:
        if value not in axis_options:
            axis_options.append(value)

    for graph_name, line_index in branch_keys:
        graph = _graph_for_key(context, graph_name)
        polyline = graph.polylines_by_index.get(int(line_index))
        if polyline is None:
            continue

        base_projection = graph.nearest_point(baseline_contact)
        if base_projection is not None and int(base_projection.line_index) == int(line_index):
            base_arclength_mm = float(base_projection.line_arclength_mm)
        else:
            markup_projection = graph.nearest_point(np.asarray(pose.contact_world, dtype=np.float64))
            if markup_projection is not None and int(markup_projection.line_index) == int(line_index):
                base_arclength_mm = float(markup_projection.line_arclength_mm)
            else:
                base_arclength_mm = float(polyline.total_length_mm / 2.0)

        for branch_shift_mm in FLAGGED_BRANCH_SHIFT_MM:
            candidate_s = float(np.clip(base_arclength_mm + float(branch_shift_mm), 0.0, polyline.total_length_mm))
            contact_seed_world = polyline.point_at_arc_length(candidate_s)
            shaft_axis_world = graph.estimate_tangent(line_index=int(line_index), line_arclength_mm=candidate_s)
            shaft_axis_world = _normalize(shaft_axis_world) if shaft_axis_world is not None else None
            if shaft_axis_world is None:
                continue

            candidate_depth_axis = _derive_local_candidate_depth_axis(
                contact_seed_world=contact_seed_world,
                target_world=target_world,
                shaft_axis_world=shaft_axis_world,
                fallback_probe_axis_world=fallback_probe_axis,
            )

            for roll_delta_deg in FLAGGED_ROLL_DELTA_DEG:
                candidate_roll_offset_deg = float(base_roll_offset_deg + float(roll_delta_deg))
                for axis_sign_override in axis_options:
                    candidate_device_pose = build_device_pose(
                        pose,
                        device_name=device,
                        ct_volume=context.ct_volume,
                        airway_lumen=context.airway_lumen_volume,
                        airway_solid=context.airway_solid_volume,
                        raw_airway_mesh=context.airway_geometry_mesh,
                        main_graph=context.main_graph,
                        network_graph=context.network_graph,
                        refine_contact=True,
                        roll_offset_deg=candidate_roll_offset_deg,
                        axis_sign_override=axis_sign_override,
                        branch_hint=branch_hint,
                        contact_seed_world=contact_seed_world,
                        shaft_axis_override=shaft_axis_world,
                        depth_axis_override=candidate_depth_axis,
                    )
                    candidate_metrics = compute_pose_review_metrics(
                        context,
                        preset_manifest=preset_manifest,
                        target_world=np.asarray(candidate_device_pose.target_world, dtype=np.float64),
                        contact_world=np.asarray(candidate_device_pose.contact_refinement.refined_contact_world, dtype=np.float64),
                        probe_axis=np.asarray(candidate_device_pose.probe_axis_world, dtype=np.float64),
                        shaft_axis=np.asarray(candidate_device_pose.shaft_axis_world, dtype=np.float64),
                        thickness_axis=np.asarray(candidate_device_pose.lateral_axis_world, dtype=np.float64),
                        warnings=list(candidate_device_pose.contact_refinement.warnings),
                        width=width,
                        height=height,
                        source_oblique_size_mm=source_oblique_size_mm,
                        max_depth_mm=max_depth_mm,
                        sector_angle_deg=sector_angle_deg,
                        slice_thickness_mm=slice_thickness_mm,
                        nUS_delta_deg_from_voxel_baseline=_angle_deg(
                            np.asarray(candidate_device_pose.voxel_probe_axis_world, dtype=np.float64),
                            np.asarray(candidate_device_pose.probe_axis_world, dtype=np.float64),
                        ),
                        contact_delta_mm_from_voxel_baseline=float(candidate_device_pose.contact_refinement.voxel_to_mesh_contact_distance_mm),
                    )
                    objective = _local_pose_objective(
                        metrics=candidate_metrics,
                        device_pose=candidate_device_pose,
                        baseline_device_pose=base_device_pose,
                        branch_shift_mm=float(branch_shift_mm),
                        roll_offset_deg=candidate_roll_offset_deg,
                        base_roll_offset_deg=float(base_roll_offset_deg),
                        axis_sign_override=axis_sign_override,
                        base_axis_sign_override=base_axis_sign_override,
                    )
                    if objective > best.objective:
                        best = LocalPoseOptimizationResult(
                            device_pose=candidate_device_pose,
                            branch_shift_mm=float(branch_shift_mm),
                            roll_offset_deg=candidate_roll_offset_deg,
                            axis_sign_override=axis_sign_override,
                            objective=objective,
                            metrics=candidate_metrics,
                        )

    return best

def _sample_mask_presence(
    context: RenderContext,
    *,
    mask_path: Path,
    base_points_lps: np.ndarray,
    thickness_axis: np.ndarray,
    slice_thickness_mm: float,
) -> np.ndarray:
    mask_volume = _get_mask_volume(context, mask_path)
    return _sample_slab(
        np.asarray(mask_volume.data, dtype=np.float32),
        base_points_lps=base_points_lps,
        thickness_axis=thickness_axis,
        inverse_affine_lps=mask_volume.inverse_affine_lps,
        sample_count=DEFAULT_SLAB_SAMPLES,
        slab_thickness_mm=slice_thickness_mm,
        order=0,
        cval=0.0,
    )


def _default_output_stem(preset_id: str, approach: str) -> str:
    return preset_id if approach == "default" else f"{preset_id}_{approach}"


def _build_sector_render(
    context: RenderContext,
    *,
    pose,
    preset_manifest,
    width: int,
    height: int,
    sector_angle_deg: float,
    max_depth_mm: float,
    gain: float,
    attenuation: float,
    slice_thickness_mm: float,
    overlay_config: OverlayConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    depth_grid, lateral_grid, sector_mask, max_lateral_mm = _build_sector_grid(width, height, max_depth_mm, sector_angle_deg)

    contact_world = np.asarray(pose.contact_world, dtype=np.float64)
    depth_axis = np.asarray(pose.depth_axis, dtype=np.float64)
    lateral_axis = np.asarray(pose.lateral_axis, dtype=np.float64)
    shaft_axis = np.asarray(pose.shaft_axis, dtype=np.float64)

    sector_depths = depth_grid[sector_mask]
    sector_laterals = lateral_grid[sector_mask]
    base_points_lps = (
        contact_world[None, :]
        + sector_depths[:, None] * depth_axis[None, :]
        + sector_laterals[:, None] * lateral_axis[None, :]
    )

    ct_samples = _sample_slab(
        np.asarray(context.ct_volume.data, dtype=np.float32),
        base_points_lps=base_points_lps,
        thickness_axis=shaft_axis,
        inverse_affine_lps=context.ct_volume.inverse_affine_lps,
        sample_count=DEFAULT_SLAB_SAMPLES,
        slab_thickness_mm=slice_thickness_mm,
        order=1,
        cval=-1000.0,
    )
    ct_intensity = _window_ct(
        ct_samples,
        gain=gain,
        attenuation=attenuation,
        depths_mm=sector_depths,
        max_depth_mm=max_depth_mm,
    )

    clean = np.zeros((height, width, 3), dtype=np.float32)
    clean[sector_mask] = np.repeat(ct_intensity[:, None], 3, axis=1)
    contours = clean.copy()

    if overlay_config.airway_lumen_enabled:
        lumen_samples = _sample_mask_presence(
            context,
            mask_path=context.manifest.airway_lumen_mask,
            base_points_lps=base_points_lps,
            thickness_axis=shaft_axis,
            slice_thickness_mm=slice_thickness_mm,
        )
        lumen_mask = np.zeros((height, width), dtype=bool)
        lumen_mask[sector_mask] = lumen_samples > 0.0
        _apply_contour_overlay(contours, lumen_mask, AIRWAY_LUMEN_COLOR)

    if overlay_config.airway_wall_enabled:
        wall_samples = _sample_mask_presence(
            context,
            mask_path=context.manifest.airway_solid_mask,
            base_points_lps=base_points_lps,
            thickness_axis=shaft_axis,
            slice_thickness_mm=slice_thickness_mm,
        )
        wall_mask = np.zeros((height, width), dtype=bool)
        wall_mask[sector_mask] = wall_samples > 0.0
        _apply_contour_overlay(contours, wall_mask, AIRWAY_WALL_COLOR)

    if overlay_config.station_enabled:
        station_samples = _sample_mask_presence(
            context,
            mask_path=preset_manifest.station_mask,
            base_points_lps=base_points_lps,
            thickness_axis=shaft_axis,
            slice_thickness_mm=slice_thickness_mm,
        )
        station_mask = np.zeros((height, width), dtype=bool)
        station_mask[sector_mask] = station_samples > 0.0
        _apply_contour_overlay(contours, station_mask, STATION_COLOR)

    for index, vessel_name in enumerate(overlay_config.vessel_names):
        vessel_samples = _sample_mask_presence(
            context,
            mask_path=context.manifest.overlay_masks[vessel_name],
            base_points_lps=base_points_lps,
            thickness_axis=shaft_axis,
            slice_thickness_mm=slice_thickness_mm,
        )
        vessel_mask = np.zeros((height, width), dtype=bool)
        vessel_mask[sector_mask] = vessel_samples > 0.0
        vessel_color = VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)]
        _apply_contour_overlay(contours, vessel_mask, vessel_color)

    if overlay_config.target_enabled:
        target_world = np.asarray(pose.target_world, dtype=np.float64)
        target_offset = target_world - contact_world
        target_depth_mm = float(np.dot(target_offset, depth_axis))
        target_lateral_mm = float(np.dot(target_offset, lateral_axis))
        if 0.0 <= target_depth_mm <= max_depth_mm and abs(target_lateral_mm) <= (target_depth_mm * np.tan(np.deg2rad(sector_angle_deg / 2.0))):
            target_row = int(round((target_depth_mm / max(max_depth_mm, 1e-6)) * (height - 1)))
            target_column = int(round((target_lateral_mm / max(max_lateral_mm, 1e-6)) * max(1, width // 2) + (width // 2)))
            _draw_cross_marker(contours, row=target_row, column=target_column, color_rgb=TARGET_MARKER_COLOR, radius=4)

    if overlay_config.contact_enabled:
        _draw_cross_marker(contours, row=0, column=width // 2, color_rgb=CONTACT_MARKER_COLOR, radius=4)

    return clean, contours, sector_mask, base_points_lps, max_lateral_mm


def _build_reference_slice(
    context: RenderContext,
    *,
    pose,
    preset_manifest,
    width: int,
    height: int,
    max_depth_mm: float,
    slice_thickness_mm: float,
    overlay_config: OverlayConfig,
) -> np.ndarray:
    contact_world = np.asarray(pose.contact_world, dtype=np.float64)
    target_world = np.asarray(pose.target_world, dtype=np.float64)
    depth_axis = np.asarray(pose.depth_axis, dtype=np.float64)
    lateral_axis = np.asarray(pose.lateral_axis, dtype=np.float64)
    shaft_axis = np.asarray(pose.shaft_axis, dtype=np.float64)

    target_offset = target_world - contact_world
    target_depth_mm = float(np.dot(target_offset, depth_axis))
    target_shaft_mm = float(np.dot(target_offset, shaft_axis))

    x_min_mm = -REFERENCE_SLICE_DEPTH_PADDING_MM
    x_max_mm = max(max_depth_mm, target_depth_mm + REFERENCE_SLICE_DEPTH_PADDING_MM)
    y_min_mm = min(-REFERENCE_SLICE_SHAFT_HALF_SPAN_MM, target_shaft_mm - 8.0)
    y_max_mm = max(REFERENCE_SLICE_SHAFT_HALF_SPAN_MM, target_shaft_mm + 8.0)

    ct_samples, _, _, _ = _sample_plane(
        context,
        x_axis=depth_axis,
        y_axis=shaft_axis,
        thickness_axis=lateral_axis,
        center_world=contact_world,
        x_min_mm=x_min_mm,
        x_max_mm=x_max_mm,
        y_min_mm=y_min_mm,
        y_max_mm=y_max_mm,
        width=width,
        height=height,
        slice_thickness_mm=slice_thickness_mm,
        order=1,
        cval=-1000.0,
        data=np.asarray(context.ct_volume.data, dtype=np.float32),
        inverse_affine_lps=context.ct_volume.inverse_affine_lps,
    )
    reference = np.repeat(
        _window_ct(
            ct_samples.reshape(-1),
            gain=1.0,
            attenuation=0.0,
            depths_mm=np.zeros(width * height, dtype=np.float32),
            max_depth_mm=max_depth_mm,
        ).reshape((height, width))[:, :, None],
        3,
        axis=2,
    )

    def _overlay_plane_mask(mask_path: Path, color: np.ndarray) -> None:
        mask_volume = _get_mask_volume(context, mask_path)
        samples, _, _, _ = _sample_plane(
            context,
            x_axis=depth_axis,
            y_axis=shaft_axis,
            thickness_axis=lateral_axis,
            center_world=contact_world,
            x_min_mm=x_min_mm,
            x_max_mm=x_max_mm,
            y_min_mm=y_min_mm,
            y_max_mm=y_max_mm,
            width=width,
            height=height,
            slice_thickness_mm=slice_thickness_mm,
            order=0,
            cval=0.0,
            data=np.asarray(mask_volume.data, dtype=np.float32),
            inverse_affine_lps=mask_volume.inverse_affine_lps,
        )
        _apply_contour_overlay(reference, samples > 0.0, color)

    if overlay_config.airway_lumen_enabled:
        _overlay_plane_mask(context.manifest.airway_lumen_mask, AIRWAY_LUMEN_COLOR)
    if overlay_config.airway_wall_enabled:
        _overlay_plane_mask(context.manifest.airway_solid_mask, AIRWAY_WALL_COLOR)
    if overlay_config.station_enabled:
        _overlay_plane_mask(preset_manifest.station_mask, STATION_COLOR)
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        _overlay_plane_mask(context.manifest.overlay_masks[vessel_name], VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)])

    if overlay_config.contact_enabled:
        row, column = _plane_point_to_pixel(
            0.0,
            0.0,
            x_min_mm=x_min_mm,
            x_max_mm=x_max_mm,
            y_min_mm=y_min_mm,
            y_max_mm=y_max_mm,
            width=width,
            height=height,
        )
        _draw_cross_marker(reference, row=row, column=column, color_rgb=CONTACT_MARKER_COLOR, radius=4)

    if overlay_config.target_enabled:
        row, column = _plane_point_to_pixel(
            target_depth_mm,
            target_shaft_mm,
            x_min_mm=x_min_mm,
            x_max_mm=x_max_mm,
            y_min_mm=y_min_mm,
            y_max_mm=y_max_mm,
            width=width,
            height=height,
        )
        _draw_cross_marker(reference, row=row, column=column, color_rgb=TARGET_MARKER_COLOR, radius=4)

    return reference


def _extract_local_centerline_points(context: RenderContext, pose, contact_world: np.ndarray, depth_axis: np.ndarray, lateral_axis: np.ndarray, shaft_axis: np.ndarray) -> np.ndarray:
    query = pose.centerline_query
    if query is None or query.line_index is None or query.line_arclength_mm is None:
        return np.empty((0, 3), dtype=np.float64)

    polyline = context.main_graph.polylines_by_index.get(int(query.line_index))
    if polyline is None:
        return np.empty((0, 3), dtype=np.float64)

    lower = float(query.line_arclength_mm - CONTEXT_CENTERLINE_WINDOW_MM)
    upper = float(query.line_arclength_mm + CONTEXT_CENTERLINE_WINDOW_MM)
    mask = np.logical_and(polyline.cumulative_lengths_mm >= lower, polyline.cumulative_lengths_mm <= upper)
    points = polyline.points_lps[mask]
    if points.shape[0] < 2:
        points = np.stack(
            [
                polyline.point_at_arc_length(max(0.0, query.line_arclength_mm - CONTEXT_CENTERLINE_WINDOW_MM)),
                polyline.point_at_arc_length(query.line_arclength_mm),
                polyline.point_at_arc_length(min(polyline.total_length_mm, query.line_arclength_mm + CONTEXT_CENTERLINE_WINDOW_MM)),
            ],
            axis=0,
        )

    relative = points - contact_world[None, :]
    return np.column_stack(
        (
            relative @ lateral_axis,
            relative @ depth_axis,
            relative @ shaft_axis,
        )
    )


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int = 3) -> None:
    draw.line((start, end), fill=color, width=width)
    direction = np.asarray([end[0] - start[0], end[1] - start[1]], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-6:
        return
    direction /= norm
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    head = np.asarray(end, dtype=np.float64)
    left = head - direction * 10.0 + perpendicular * 4.0
    right = head - direction * 10.0 - perpendicular * 4.0
    draw.line((tuple(head), tuple(left)), fill=color, width=width)
    draw.line((tuple(head), tuple(right)), fill=color, width=width)


def _build_context_snapshot(
    context: RenderContext,
    *,
    pose,
    width: int,
    height: int,
) -> np.ndarray:
    contact_world = np.asarray(pose.contact_world, dtype=np.float64)
    target_world = np.asarray(pose.target_world, dtype=np.float64)
    depth_axis = np.asarray(pose.depth_axis, dtype=np.float64)
    lateral_axis = np.asarray(pose.lateral_axis, dtype=np.float64)
    shaft_axis = np.asarray(pose.shaft_axis, dtype=np.float64)

    target_offset = target_world - contact_world
    target_local = np.asarray(
        [
            float(np.dot(target_offset, lateral_axis)),
            float(np.dot(target_offset, depth_axis)),
            float(np.dot(target_offset, shaft_axis)),
        ],
        dtype=np.float64,
    )
    centerline_local = _extract_local_centerline_points(context, pose, contact_world, depth_axis, lateral_axis, shaft_axis)

    axis_endpoints = np.asarray(
        [
            [0.0, CONTEXT_AXIS_LENGTH_MM, 0.0],
            [CONTEXT_AXIS_LENGTH_MM, 0.0, 0.0],
            [0.0, 0.0, CONTEXT_AXIS_LENGTH_MM],
            target_local,
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    if centerline_local.size:
        cloud = np.vstack((axis_endpoints, centerline_local))
    else:
        cloud = axis_endpoints

    projected = np.column_stack(
        (
            cloud[:, 0] * 0.92 + cloud[:, 1] * 0.35,
            -cloud[:, 2] * 0.82 + cloud[:, 1] * 0.24,
        )
    )
    min_xy = projected.min(axis=0)
    max_xy = projected.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    margin = 22.0
    scale = min((width - 2.0 * margin) / span[0], (height - 2.0 * margin) / span[1])
    center_xy = (min_xy + max_xy) / 2.0

    def _project_local(local: np.ndarray) -> tuple[float, float]:
        coords = np.asarray(
            [
                local[0] * 0.92 + local[1] * 0.35,
                -local[2] * 0.82 + local[1] * 0.24,
            ],
            dtype=np.float64,
        )
        centered = (coords - center_xy) * scale
        return float((width / 2.0) + centered[0]), float((height / 2.0) + centered[1])

    image = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(image)

    if centerline_local.shape[0] >= 2:
        centerline_points = [_project_local(point) for point in centerline_local]
        draw.line(centerline_points, fill=CENTERLINE_CONTEXT_COLOR, width=3)

    contact_xy = _project_local(np.asarray([0.0, 0.0, 0.0], dtype=np.float64))
    target_xy = _project_local(target_local)
    draw.line((contact_xy, target_xy), fill=(230, 190, 80), width=2)

    depth_xy = _project_local(np.asarray([0.0, CONTEXT_AXIS_LENGTH_MM, 0.0], dtype=np.float64))
    lateral_xy = _project_local(np.asarray([CONTEXT_AXIS_LENGTH_MM, 0.0, 0.0], dtype=np.float64))
    shaft_xy = _project_local(np.asarray([0.0, 0.0, CONTEXT_AXIS_LENGTH_MM], dtype=np.float64))

    _arrow(draw, contact_xy, depth_xy, (255, 120, 120))
    _arrow(draw, contact_xy, lateral_xy, (120, 215, 255))
    _arrow(draw, contact_xy, shaft_xy, (180, 255, 160))
    draw.text((depth_xy[0] + 4, depth_xy[1] - 10), "depth", fill=(255, 120, 120))
    draw.text((lateral_xy[0] + 4, lateral_xy[1] - 10), "lateral", fill=(120, 215, 255))
    draw.text((shaft_xy[0] + 4, shaft_xy[1] - 10), "shaft", fill=(180, 255, 160))

    draw.ellipse((contact_xy[0] - 4, contact_xy[1] - 4, contact_xy[0] + 4, contact_xy[1] + 4), fill=(255, 235, 140))
    draw.text((contact_xy[0] + 6, contact_xy[1] + 4), "contact", fill=(255, 235, 140))
    draw.ellipse((target_xy[0] - 4, target_xy[1] - 4, target_xy[0] + 4, target_xy[1] + 4), fill=(255, 85, 85))
    draw.text((target_xy[0] + 6, target_xy[1] + 4), "target", fill=(255, 85, 85))

    return np.asarray(image, dtype=np.uint8)


def _extract_local_mask_points(
    volume: VolumeData,
    *,
    center_world: np.ndarray,
    lateral_axis: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    radius_mm: float,
    max_points: int,
) -> np.ndarray:
    if volume.data is None:
        return np.empty((0, 3), dtype=np.float64)
    center_ijk = _points_to_voxel(center_world[None, :], volume.inverse_affine_lps)[0]
    voxel_radius = np.maximum(1.0, np.ceil(radius_mm / np.maximum(volume.voxel_sizes_mm[:3], 1e-6))).astype(int)
    lower = np.maximum(0, np.floor(center_ijk).astype(int) - voxel_radius)
    upper = np.minimum(np.asarray(volume.shape[:3], dtype=int), np.floor(center_ijk).astype(int) + voxel_radius + 1)
    crop = np.asarray(volume.data[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] > 0, dtype=bool)
    if not np.any(crop):
        return np.empty((0, 3), dtype=np.float64)

    surface = np.logical_and(crop, np.logical_not(ndimage.binary_erosion(crop, border_value=0)))
    indices = np.argwhere(surface)
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    stride = max(1, int(np.ceil(indices.shape[0] / max_points)))
    indices = indices[::stride] + lower[None, :]
    homogeneous = np.concatenate((indices.astype(np.float64), np.ones((indices.shape[0], 1), dtype=np.float64)), axis=1)
    world_points = homogeneous @ volume.affine_lps.T
    relative = world_points[:, :3] - center_world[None, :]
    return np.column_stack((relative @ lateral_axis, relative @ probe_axis, relative @ shaft_axis))


def _context_projected_xy(local_points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            local_points @ CONTEXT_SCREEN_X_BASIS,
            local_points @ CONTEXT_SCREEN_Y_BASIS,
        )
    )


def _transform_world_triangles_to_local(
    triangles_world: np.ndarray,
    *,
    contact_world: np.ndarray,
    lateral_axis: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
) -> np.ndarray:
    if triangles_world.size == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    relative = triangles_world - contact_world[None, None, :]
    return np.stack(
        (
            relative @ lateral_axis,
            relative @ probe_axis,
            relative @ shaft_axis,
        ),
        axis=2,
    )


def _triangle_area_screen(points_xy: list[tuple[float, float]]) -> float:
    ax, ay = points_xy[0]
    bx, by = points_xy[1]
    cx, cy = points_xy[2]
    return abs(((bx - ax) * (cy - ay)) - ((cx - ax) * (by - ay))) * 0.5


def _draw_context_mesh(
    draw: ImageDraw.ImageDraw,
    *,
    triangles_local: np.ndarray,
    project_local,
    base_color_rgb: np.ndarray,
    plane_origin_local: np.ndarray,
    plane_normal_local: np.ndarray,
    show_full_airway: bool,
) -> None:
    if triangles_local.size == 0:
        return

    base_color = np.asarray(base_color_rgb, dtype=np.float64)
    view_direction = CONTEXT_VIEW_DIRECTION / max(float(np.linalg.norm(CONTEXT_VIEW_DIRECTION)), 1e-6)
    depth_order = np.argsort(triangles_local.mean(axis=1) @ view_direction)

    for triangle_index in depth_order.tolist():
        triangle_local = triangles_local[triangle_index]
        face_normal = np.cross(triangle_local[1] - triangle_local[0], triangle_local[2] - triangle_local[0])
        face_normal_norm = float(np.linalg.norm(face_normal))
        if face_normal_norm <= 1e-6:
            continue

        projected = [project_local(vertex) for vertex in triangle_local]
        if _triangle_area_screen(projected) <= 0.05:
            continue

        face_normal /= face_normal_norm
        shading = 0.34 + (0.40 * abs(float(np.dot(face_normal, view_direction))))
        fill = tuple(
            int(np.clip(channel * shading * 255.0, 0.0, 255.0))
            for channel in base_color.tolist()
        )
        draw.polygon(projected, fill=fill)

        if show_full_airway:
            continue

        signed = (triangle_local - plane_origin_local[None, :]) @ plane_normal_local
        if np.count_nonzero(np.abs(signed) <= CUTAWAY_TRIANGLE_EPSILON_MM) >= 2:
            draw.line((projected[0], projected[1]), fill=CUTAWAY_CONTEXT_EDGE_COLOR, width=1)
            draw.line((projected[1], projected[2]), fill=CUTAWAY_CONTEXT_EDGE_COLOR, width=1)
            draw.line((projected[2], projected[0]), fill=CUTAWAY_CONTEXT_EDGE_COLOR, width=1)


def _build_cp_context_snapshot(
    context: RenderContext,
    *,
    pose,
    device_pose: DevicePose,
    overlay_config: OverlayConfig,
    cutaway_display: CutawayDisplay,
    station_local: np.ndarray,
    width: int,
    height: int,
    max_depth_mm: float,
) -> np.ndarray:
    contact_world = np.asarray(device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    original_contact_world = np.asarray(device_pose.contact_refinement.original_contact_world, dtype=np.float64)
    target_world = np.asarray(device_pose.target_world, dtype=np.float64)
    shaft_axis = np.asarray(device_pose.shaft_axis_world, dtype=np.float64)
    probe_axis = np.asarray(device_pose.probe_axis_world, dtype=np.float64)
    lateral_axis = np.asarray(device_pose.lateral_axis_world, dtype=np.float64)
    video_axis = np.asarray(device_pose.video_axis_world, dtype=np.float64)
    target_offset = target_world - contact_world
    target_local = np.asarray(
        [
            float(np.dot(target_offset, lateral_axis)),
            float(np.dot(target_offset, probe_axis)),
            float(np.dot(target_offset, shaft_axis)),
        ],
        dtype=np.float64,
    )
    original_offset = original_contact_world - contact_world
    original_local = np.asarray(
        [
            float(np.dot(original_offset, lateral_axis)),
            float(np.dot(original_offset, probe_axis)),
            float(np.dot(original_offset, shaft_axis)),
        ],
        dtype=np.float64,
    )
    centerline_local = _extract_local_centerline_points(context, pose, contact_world, probe_axis, lateral_axis, shaft_axis)
    airway_triangles_local = _transform_world_triangles_to_local(
        cutaway_display.triangles_world,
        contact_world=contact_world,
        lateral_axis=lateral_axis,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
    )
    airway_fallback_local = _extract_local_mask_points(
        context.airway_solid_volume,
        center_world=contact_world,
        lateral_axis=lateral_axis,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        radius_mm=32.0,
        max_points=1400,
    )
    cutaway_origin_offset = cutaway_display.origin_world - contact_world
    cutaway_origin_local = np.asarray(
        [
            float(np.dot(cutaway_origin_offset, lateral_axis)),
            float(np.dot(cutaway_origin_offset, probe_axis)),
            float(np.dot(cutaway_origin_offset, shaft_axis)),
        ],
        dtype=np.float64,
    )
    cutaway_normal_local = np.asarray(
        [
            float(np.dot(cutaway_display.normal_world, lateral_axis)),
            float(np.dot(cutaway_display.normal_world, probe_axis)),
            float(np.dot(cutaway_display.normal_world, shaft_axis)),
        ],
        dtype=np.float64,
    )
    vessel_clouds: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        vessel_points = _extract_local_mask_points(
            _get_mask_volume(context, context.manifest.overlay_masks[vessel_name]),
            center_world=contact_world,
            lateral_axis=lateral_axis,
            probe_axis=probe_axis,
            shaft_axis=shaft_axis,
            radius_mm=32.0,
            max_points=900,
        )
        if vessel_points.size:
            vessel_clouds.append((vessel_points, _color_tuple(VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)])))

    half_angle = np.deg2rad(float(device_pose.device_model.sector_angle_deg) / 2.0)
    left_frustum = np.asarray([0.0, max_depth_mm * np.cos(half_angle), -max_depth_mm * np.sin(half_angle)], dtype=np.float64)
    right_frustum = np.asarray([0.0, max_depth_mm * np.cos(half_angle), max_depth_mm * np.sin(half_angle)], dtype=np.float64)
    probe_axis_local = np.asarray([0.0, CONTEXT_AXIS_LENGTH_MM, 0.0], dtype=np.float64)
    shaft_axis_local = np.asarray([0.0, 0.0, CONTEXT_AXIS_LENGTH_MM], dtype=np.float64)
    video_axis_local = np.asarray(
        [
            float(np.dot(video_axis, lateral_axis)),
            float(np.dot(video_axis, probe_axis)),
            float(np.dot(video_axis, shaft_axis)),
        ],
        dtype=np.float64,
    ) * CONTEXT_AXIS_LENGTH_MM

    cloud_parts = [
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                target_local,
                left_frustum,
                right_frustum,
                probe_axis_local,
                shaft_axis_local,
                video_axis_local,
            ],
            dtype=np.float64,
        ),
        centerline_local if centerline_local.size else np.empty((0, 3), dtype=np.float64),
    ]
    if airway_triangles_local.size:
        cloud_parts.append(airway_triangles_local.reshape((-1, 3)))
    elif airway_fallback_local.size:
        cloud_parts.append(airway_fallback_local)
    if overlay_config.station_enabled and station_local.size:
        cloud_parts.append(station_local)
    cloud_parts.extend(points for points, _ in vessel_clouds if points.size)
    cloud = np.vstack(cloud_parts)
    projected = _context_projected_xy(cloud)
    min_xy = projected.min(axis=0)
    max_xy = projected.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    margin = 22.0
    scale = min((width - 2.0 * margin) / span[0], (height - 2.0 * margin) / span[1])
    center_xy = (min_xy + max_xy) / 2.0

    def _project_local(local: np.ndarray) -> tuple[float, float]:
        coords = _context_projected_xy(np.asarray(local, dtype=np.float64)[None, :])[0]
        centered = (coords - center_xy) * scale
        return float((width / 2.0) + centered[0]), float((height / 2.0) + centered[1])

    image = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(image)

    if airway_triangles_local.size:
        _draw_context_mesh(
            draw,
            triangles_local=airway_triangles_local,
            project_local=_project_local,
            base_color_rgb=AIRWAY_WALL_COLOR,
            plane_origin_local=cutaway_origin_local,
            plane_normal_local=cutaway_normal_local,
            show_full_airway=cutaway_display.show_full_airway,
        )
    else:
        for point in airway_fallback_local:
            x, y = _project_local(point)
            draw.point((x, y), fill=_color_tuple(AIRWAY_WALL_COLOR))

    if overlay_config.station_enabled:
        for point in station_local:
            x, y = _project_local(point)
            draw.point((x, y), fill=_color_tuple(STATION_COLOR))
    for vessel_points, vessel_color in vessel_clouds:
        for point in vessel_points:
            x, y = _project_local(point)
            draw.point((x, y), fill=vessel_color)

    if centerline_local.shape[0] >= 2:
        draw.line([_project_local(point) for point in centerline_local], fill=CENTERLINE_CONTEXT_COLOR, width=3)

    contact_xy = _project_local(np.asarray([0.0, 0.0, 0.0], dtype=np.float64))
    target_xy = _project_local(target_local)
    original_xy = _project_local(original_local)
    probe_xy = _project_local(probe_axis_local)
    shaft_xy = _project_local(shaft_axis_local)
    video_xy = _project_local(video_axis_local)
    left_xy = _project_local(left_frustum)
    right_xy = _project_local(right_frustum)

    if overlay_config.show_frustum:
        draw.line((contact_xy, left_xy), fill=FRUSTUM_CONTEXT_COLOR, width=2)
        draw.line((contact_xy, right_xy), fill=FRUSTUM_CONTEXT_COLOR, width=2)
        draw.line((left_xy, right_xy), fill=FRUSTUM_CONTEXT_COLOR, width=2)

    draw.line((contact_xy, target_xy), fill=(230, 190, 80), width=2)
    _arrow(draw, contact_xy, probe_xy, PROBE_AXIS_CONTEXT_COLOR)
    _arrow(draw, contact_xy, shaft_xy, (180, 255, 160))
    _arrow(draw, contact_xy, video_xy, VIDEO_AXIS_CONTEXT_COLOR)
    draw.text((probe_xy[0] + 4, probe_xy[1] - 10), "nUS", fill=PROBE_AXIS_CONTEXT_COLOR)
    draw.text((shaft_xy[0] + 4, shaft_xy[1] - 10), "nB", fill=(180, 255, 160))
    draw.text((video_xy[0] + 4, video_xy[1] - 10), "nC", fill=VIDEO_AXIS_CONTEXT_COLOR)

    draw.ellipse((contact_xy[0] - 4, contact_xy[1] - 4, contact_xy[0] + 4, contact_xy[1] + 4), fill=(255, 235, 140))
    draw.text((contact_xy[0] + 6, contact_xy[1] + 4), "refined contact", fill=(255, 235, 140))
    if np.linalg.norm(original_local) > 1e-3:
        draw.ellipse((original_xy[0] - 3, original_xy[1] - 3, original_xy[0] + 3, original_xy[1] + 3), outline=(190, 190, 190))
        draw.text((original_xy[0] + 6, original_xy[1] + 2), "markup", fill=(190, 190, 190))
    draw.ellipse((target_xy[0] - 4, target_xy[1] - 4, target_xy[0] + 4, target_xy[1] + 4), fill=(255, 85, 85))
    draw.text((target_xy[0] + 6, target_xy[1] + 4), "target", fill=(255, 85, 85))
    if overlay_config.station_enabled and station_local.size:
        station_center = station_local.mean(axis=0)
        station_xy = _project_local(station_center)
        draw.text((station_xy[0] + 6, station_xy[1] - 10), f"station {pose.station.upper()}", fill=_color_tuple(STATION_COLOR))

    return np.asarray(image, dtype=np.uint8)


def _compose_cp_diagnostic_panel(virtual_view: np.ndarray, simulated_view: np.ndarray, localizer_view: np.ndarray, context_snapshot: np.ndarray) -> np.ndarray:
    height, width = virtual_view.shape[:2]
    virtual_tile = Image.fromarray(_add_panel_label(virtual_view.astype(np.float32) / 255.0, "Virtual EBUS"), mode="RGB")
    simulated_tile = Image.fromarray(_add_panel_label(simulated_view.astype(np.float32) / 255.0, "Simulated EBUS"), mode="RGB")
    localizer_tile = Image.fromarray(_add_panel_label(localizer_view.astype(np.float32) / 255.0, "Wide CT Localizer"), mode="RGB")
    context_tile = Image.fromarray(_add_panel_label(context_snapshot.astype(np.float32) / 255.0, "3D Context"), mode="RGB")

    panel = Image.new("RGB", (width * 2, height * 2), (0, 0, 0))
    panel.paste(virtual_tile, (0, 0))
    panel.paste(simulated_tile, (width, 0))
    panel.paste(localizer_tile, (0, height))
    panel.paste(context_tile, (width, height))
    return np.asarray(panel, dtype=np.uint8)


def _render_preset_localizer(
    manifest_path: str | Path,
    preset_id: str,
    *,
    approach: str | None = None,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    sector_angle_deg: float | None = None,
    max_depth_mm: float | None = None,
    roll_deg: float | None = None,
    mode: str | None = None,
    airway_overlay: bool | None = None,
    airway_lumen_overlay: bool | None = None,
    airway_wall_overlay: bool | None = None,
    target_overlay: bool | None = None,
    contact_overlay: bool | None = None,
    station_overlay: bool | None = None,
    vessel_overlay_names: list[str] | None = None,
    slice_thickness_mm: float | None = None,
    diagnostic_panel: bool = False,
    device: str = DEFAULT_DEVICE_NAME,
    refine_contact: bool = True,
    virtual_ebus: bool = True,
    simulated_ebus: bool = True,
    reference_fov_mm: float | None = None,
    source_oblique_size_mm: float | None = None,
    single_vessel: str | None = None,
    show_legend: bool | None = None,
    label_overlays: bool | None = None,
    min_contour_area_px: float = 20.0,
    min_contour_length_px: float = 15.0,
    show_contact: bool | None = None,
    show_frustum: bool | None = None,
    cutaway_mode: str | None = None,
    cutaway_side: str | None = None,
    cutaway_depth_mm: float | None = None,
    cutaway_origin: str | None = None,
    show_full_airway: bool | None = None,
    cutaway_custom_origin_world: np.ndarray | None = None,
    debug_map_dir: str | Path | None = None,
    physics_profile: str | None = None,
    speckle_strength: float | None = None,
    reverberation_strength: float | None = None,
    shadow_strength: float | None = None,
    seed: int | None = None,
    context: RenderContext | None = None,
) -> RenderedPreset:
    from ebus_simulator.localizer_renderer import render_localizer_preset

    request = RenderRequest(
        manifest_path=manifest_path,
        preset_id=preset_id,
        output_path=output_path,
        approach=approach,
        metadata_path=metadata_path,
        engine=RenderEngine.LOCALIZER,
        seed=seed,
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode=mode,
        airway_overlay=airway_overlay,
        airway_lumen_overlay=airway_lumen_overlay,
        airway_wall_overlay=airway_wall_overlay,
        target_overlay=target_overlay,
        contact_overlay=contact_overlay,
        station_overlay=station_overlay,
        vessel_overlay_names=vessel_overlay_names,
        slice_thickness_mm=slice_thickness_mm,
        diagnostic_panel=diagnostic_panel,
        device=device,
        refine_contact=refine_contact,
        virtual_ebus=virtual_ebus,
        simulated_ebus=simulated_ebus,
        reference_fov_mm=reference_fov_mm,
        source_oblique_size_mm=source_oblique_size_mm,
        single_vessel=single_vessel,
        show_legend=show_legend,
        label_overlays=label_overlays,
        min_contour_area_px=min_contour_area_px,
        min_contour_length_px=min_contour_length_px,
        show_contact=show_contact,
        show_frustum=show_frustum,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        cutaway_custom_origin_world=cutaway_custom_origin_world,
        debug_map_dir=debug_map_dir,
        physics_profile=physics_profile,
        speckle_strength=speckle_strength,
        reverberation_strength=reverberation_strength,
        shadow_strength=shadow_strength,
    )
    return render_localizer_preset(request, context=context).rendered_preset


def dispatch_render_request(
    request: RenderRequest,
    *,
    context: RenderContext | None = None,
) -> RenderResult:
    engine = parse_render_engine(request.engine)
    if engine is RenderEngine.LOCALIZER:
        from ebus_simulator.localizer_renderer import render_localizer_preset

        return render_localizer_preset(request, context=context)
    if engine is RenderEngine.PHYSICS:
        from ebus_simulator.physics_renderer import render_physics_preset

        return render_physics_preset(request, context=context)
    raise ValueError(f"Unsupported render engine {engine.value!r}.")


def render_preset(
    manifest_path: str | Path,
    preset_id: str,
    *,
    approach: str | None = None,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    engine: str | RenderEngine | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    sector_angle_deg: float | None = None,
    max_depth_mm: float | None = None,
    roll_deg: float | None = None,
    mode: str | None = None,
    airway_overlay: bool | None = None,
    airway_lumen_overlay: bool | None = None,
    airway_wall_overlay: bool | None = None,
    target_overlay: bool | None = None,
    contact_overlay: bool | None = None,
    station_overlay: bool | None = None,
    vessel_overlay_names: list[str] | None = None,
    slice_thickness_mm: float | None = None,
    diagnostic_panel: bool = False,
    device: str = DEFAULT_DEVICE_NAME,
    refine_contact: bool = True,
    virtual_ebus: bool = True,
    simulated_ebus: bool = True,
    reference_fov_mm: float | None = None,
    source_oblique_size_mm: float | None = None,
    single_vessel: str | None = None,
    show_legend: bool | None = None,
    label_overlays: bool | None = None,
    min_contour_area_px: float = 20.0,
    min_contour_length_px: float = 15.0,
    show_contact: bool | None = None,
    show_frustum: bool | None = None,
    cutaway_mode: str | None = None,
    cutaway_side: str | None = None,
    cutaway_depth_mm: float | None = None,
    cutaway_origin: str | None = None,
    show_full_airway: bool | None = None,
    cutaway_custom_origin_world: np.ndarray | None = None,
    debug_map_dir: str | Path | None = None,
    physics_profile: str | None = None,
    speckle_strength: float | None = None,
    reverberation_strength: float | None = None,
    shadow_strength: float | None = None,
    context: RenderContext | None = None,
) -> RenderedPreset:
    request = RenderRequest(
        manifest_path=manifest_path,
        preset_id=preset_id,
        output_path=output_path,
        approach=approach,
        metadata_path=metadata_path,
        engine=parse_render_engine(engine),
        seed=seed,
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode=mode,
        airway_overlay=airway_overlay,
        airway_lumen_overlay=airway_lumen_overlay,
        airway_wall_overlay=airway_wall_overlay,
        target_overlay=target_overlay,
        contact_overlay=contact_overlay,
        station_overlay=station_overlay,
        vessel_overlay_names=vessel_overlay_names,
        slice_thickness_mm=slice_thickness_mm,
        diagnostic_panel=diagnostic_panel,
        device=device,
        refine_contact=refine_contact,
        virtual_ebus=virtual_ebus,
        simulated_ebus=simulated_ebus,
        reference_fov_mm=reference_fov_mm,
        source_oblique_size_mm=source_oblique_size_mm,
        single_vessel=single_vessel,
        show_legend=show_legend,
        label_overlays=label_overlays,
        min_contour_area_px=min_contour_area_px,
        min_contour_length_px=min_contour_length_px,
        show_contact=show_contact,
        show_frustum=show_frustum,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        cutaway_custom_origin_world=cutaway_custom_origin_world,
        debug_map_dir=debug_map_dir,
        physics_profile=physics_profile,
        speckle_strength=speckle_strength,
        reverberation_strength=reverberation_strength,
        shadow_strength=shadow_strength,
    )
    return dispatch_render_request(request, context=context).rendered_preset


def render_all_presets(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    engine: str | RenderEngine | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    sector_angle_deg: float | None = None,
    max_depth_mm: float | None = None,
    roll_deg: float | None = None,
    mode: str | None = None,
    airway_overlay: bool | None = None,
    target_overlay: bool | None = None,
    station_overlay: bool | None = None,
    vessel_overlay_names: list[str] | None = None,
    slice_thickness_mm: float | None = None,
    debug_map_dir: str | Path | None = None,
    physics_profile: str | None = None,
    speckle_strength: float | None = None,
    reverberation_strength: float | None = None,
    shadow_strength: float | None = None,
) -> BatchRenderIndex:
    context = build_render_context(manifest_path, roll_deg=roll_deg)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_engine = parse_render_engine(engine)

    entries: list[RenderIndexEntry] = []
    for preset in context.manifest.presets:
        for approach in preset.contacts:
            stem = _default_output_stem(preset.id, approach)
            output_path = output_dir / f"{stem}.png"
            rendered = render_preset(
                context.manifest.manifest_path,
                preset.id,
                approach=approach,
                output_path=output_path,
                engine=resolved_engine,
                seed=seed,
                width=width,
                height=height,
                sector_angle_deg=sector_angle_deg,
                max_depth_mm=max_depth_mm,
                roll_deg=roll_deg,
                mode=mode,
                airway_overlay=airway_overlay,
                target_overlay=target_overlay,
                contact_overlay=None,
                station_overlay=station_overlay,
                vessel_overlay_names=vessel_overlay_names,
                slice_thickness_mm=slice_thickness_mm,
                debug_map_dir=(None if debug_map_dir is None else Path(debug_map_dir).expanduser().resolve() / stem),
                physics_profile=physics_profile,
                speckle_strength=speckle_strength,
                reverberation_strength=reverberation_strength,
                shadow_strength=shadow_strength,
                context=context,
            )
            metadata = rendered.metadata
            entries.append(
                RenderIndexEntry(
                    preset_id=metadata.preset_id,
                    approach=metadata.approach,
                    mode=metadata.mode,
                    engine=metadata.engine,
                    output_image_path=metadata.output_path,
                    sidecar_path=metadata.metadata_path,
                    image_size=list(metadata.image_size),
                    sector_angle_deg=metadata.sector_angle_deg,
                    max_depth_mm=metadata.max_depth_mm,
                    roll_deg=metadata.roll_deg,
                    overlays_enabled=list(metadata.overlays_enabled),
                    airway_overlay_enabled=metadata.airway_overlay_enabled,
                    target_overlay_enabled=metadata.target_overlay_enabled,
                    station_overlay_enabled=metadata.station_overlay_enabled,
                    vessel_overlay_names=list(metadata.vessel_overlay_names),
                    warnings_count=len(metadata.warnings),
                )
            )

    index = BatchRenderIndex(
        manifest_path=str(context.manifest.manifest_path),
        case_id=context.manifest.case_id,
        output_dir=str(output_dir),
        mode=(DEFAULT_RENDER_MODE if mode is None else mode),
        engine=resolved_engine.value,
        render_count=len(entries),
        renders=entries,
    )

    json_path = output_dir / "index.json"
    csv_path = output_dir / "index.csv"
    json_path.write_text(json.dumps(asdict(index), indent=2))

    with csv_path.open("w", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=[
                "preset_id",
                "approach",
                "mode",
                "engine",
                "output_image_path",
                "sidecar_path",
                "image_width",
                "image_height",
                "sector_angle_deg",
                "max_depth_mm",
                "roll_deg",
                "overlays_enabled",
                "airway_overlay_enabled",
                "target_overlay_enabled",
                "station_overlay_enabled",
                "vessel_overlay_names",
                "warnings_count",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "preset_id": entry.preset_id,
                    "approach": entry.approach,
                    "mode": entry.mode,
                    "engine": entry.engine,
                    "output_image_path": entry.output_image_path,
                    "sidecar_path": entry.sidecar_path,
                    "image_width": entry.image_size[0],
                    "image_height": entry.image_size[1],
                    "sector_angle_deg": entry.sector_angle_deg,
                    "max_depth_mm": entry.max_depth_mm,
                    "roll_deg": entry.roll_deg,
                    "overlays_enabled": ",".join(entry.overlays_enabled),
                    "airway_overlay_enabled": entry.airway_overlay_enabled,
                    "target_overlay_enabled": entry.target_overlay_enabled,
                    "station_overlay_enabled": entry.station_overlay_enabled,
                    "vessel_overlay_names": ",".join(entry.vessel_overlay_names),
                    "warnings_count": entry.warnings_count,
                }
            )

    return index


def render_metadata_to_dict(metadata: RenderMetadata) -> dict:
    return asdict(metadata)
