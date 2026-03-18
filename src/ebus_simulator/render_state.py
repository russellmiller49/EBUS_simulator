from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ebus_simulator.centerline import CenterlineGraph
from ebus_simulator.device import DevicePose, build_device_pose, _parse_branch_hint
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.io.vtp import load_vtp_polydata
from ebus_simulator.manifest import load_case_manifest, resolve_preset_overrides
from ebus_simulator.models import CaseManifest, PolyData, VolumeData
from ebus_simulator.poses import generate_pose_report
from ebus_simulator.render_engines import RenderRequest
from ebus_simulator.transforms import _normalize, _project_perpendicular


CLEAN_SLICE_THICKNESS_MM = 4.0
DEBUG_SLICE_THICKNESS_MM = 1.5
DEFAULT_RENDER_MODE = "debug"
PREPARED_RENDER_STATE_VERSION = "render-state-v1"

_LOCAL_POSE_OPTIMIZATION_CACHE: dict[tuple[object, ...], LocalPoseOptimizationResult] = {}


@dataclass(slots=True)
class CutawayConfig:
    mode: str
    side: str
    depth_mm: float
    origin_mode: str
    show_full_airway: bool


@dataclass(slots=True)
class OverlayConfig:
    mode: str
    airway_lumen_enabled: bool
    airway_wall_enabled: bool
    station_enabled: bool
    target_enabled: bool
    contact_enabled: bool
    vessel_names: list[str]
    diagnostic_panel_enabled: bool
    virtual_ebus_enabled: bool
    simulated_ebus_enabled: bool
    show_legend: bool
    label_overlays: bool
    show_frustum: bool
    min_contour_area_px: float
    min_contour_length_px: float
    single_vessel_name: str | None


@dataclass(slots=True)
class RenderContext:
    manifest: CaseManifest
    pose_report: object
    ct_volume: VolumeData
    airway_lumen_volume: VolumeData
    airway_solid_volume: VolumeData
    airway_display_mesh: PolyData | None
    airway_geometry_mesh: PolyData | None
    mask_cache: dict[str, VolumeData]
    main_graph: CenterlineGraph
    network_graph: CenterlineGraph


@dataclass(slots=True)
class PreparedRenderState:
    state_version: str
    request: RenderRequest
    context: RenderContext
    manifest: CaseManifest
    pose: object
    preset_manifest: object
    preset_overrides: object | None
    overlay_config: OverlayConfig
    cutaway_config: CutawayConfig
    baseline_device_pose: DevicePose
    device_pose: DevicePose
    optimization_cache_key: tuple[object, ...] | None
    optimization_result: LocalPoseOptimizationResult | None
    resolved_width: int
    resolved_height: int
    resolved_sector_angle_deg: float
    resolved_max_depth_mm: float
    resolved_source_oblique_size_mm: float
    resolved_reference_fov_mm: float
    resolved_roll_deg: float
    resolved_gain: float
    resolved_attenuation: float
    resolved_slice_thickness_mm: float
    preset_roll_offset_deg: float
    configured_axis_sign_override: str | None
    configured_branch_hint: str | None
    configured_branch_shift_mm: float | None
    branch_shift_seed: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    contact_world: np.ndarray
    original_contact_world: np.ndarray
    target_world: np.ndarray
    probe_axis: np.ndarray
    shaft_axis: np.ndarray
    lateral_axis: np.ndarray


@dataclass(slots=True)
class RenderMetadata:
    manifest_path: str
    case_id: str
    preset_id: str
    approach: str
    mode: str
    output_path: str
    metadata_path: str
    engine: str
    engine_version: str
    seed: int | None
    view_kind: str
    image_size: list[int]
    device_model: str
    device_label: str
    sector_angle_deg: float
    max_depth_mm: float
    roll_deg: float
    gain: float
    attenuation: float
    slice_thickness_mm: float
    slice_samples: int
    source_oblique_size_mm: float
    reference_fov_mm: float
    source_plane: str
    display_plane: str
    reference_plane: str
    refine_contact_enabled: bool
    diagnostic_panel_enabled: bool
    diagnostic_panel_layout: list[str]
    virtual_ebus_enabled: bool
    simulated_ebus_enabled: bool
    contact_world: list[float]
    original_contact_world: list[float]
    voxel_refined_contact_world: list[float]
    refined_contact_world: list[float]
    tip_start_world: list[float]
    target_world: list[float]
    nearest_centerline_point: list[float] | None
    pose_axes: dict[str, list[float] | None]
    device_axes: dict[str, list[float] | None]
    target_in_default_forward_hemisphere: bool | None
    contact_to_airway_distance_mm: float | None
    original_contact_to_airway_distance_mm: float | None
    voxel_refined_contact_to_airway_distance_mm: float | None
    refined_contact_to_airway_distance_mm: float | None
    centerline_projection_distance_mm: float | None
    contact_refinement_method: str
    pose_comparison: dict[str, object]
    airway_overlay_enabled: bool
    airway_lumen_overlay_enabled: bool
    airway_wall_overlay_enabled: bool
    target_overlay_enabled: bool
    contact_overlay_enabled: bool
    station_overlay_enabled: bool
    vessel_overlay_names: list[str]
    single_vessel_name: str | None
    show_legend: bool
    label_overlays: bool
    show_frustum: bool
    cutaway_mode: str
    cutaway_side: str
    cutaway_side_source: str
    cutaway_open_side: str
    cutaway_depth_mm: float
    cutaway_origin_mode: str
    cutaway_origin: list[float]
    cutaway_normal: list[float]
    cutaway_mesh_source: str
    show_full_airway: bool
    overlays_enabled: list[str]
    visible_overlay_names: list[str]
    consistency_metrics: dict[str, object]
    preset_override_applied: bool
    preset_override_vessel_overlays: list[str]
    preset_override_cutaway_side: str | None
    preset_override_roll_offset_deg: float
    preset_override_branch_hint: str | None
    preset_override_branch_shift_mm: float | None
    preset_override_axis_sign_override: str | None
    preset_override_reference_fov_mm: float | None
    preset_override_notes: str | None
    warnings: list[str]
    engine_diagnostics: dict[str, object]


@dataclass(slots=True)
class RenderedPreset:
    image_rgb: np.ndarray
    sector_mask: np.ndarray
    metadata: RenderMetadata


@dataclass(slots=True)
class RenderIndexEntry:
    preset_id: str
    approach: str
    mode: str
    engine: str
    output_image_path: str
    sidecar_path: str
    image_size: list[int]
    sector_angle_deg: float
    max_depth_mm: float
    roll_deg: float
    overlays_enabled: list[str]
    airway_overlay_enabled: bool
    target_overlay_enabled: bool
    station_overlay_enabled: bool
    vessel_overlay_names: list[str]
    warnings_count: int


@dataclass(slots=True)
class BatchRenderIndex:
    manifest_path: str
    case_id: str
    output_dir: str
    mode: str
    engine: str
    render_count: int
    renders: list[RenderIndexEntry]


@dataclass(slots=True)
class OverlayLayer:
    key: str
    label: str
    color_rgb: np.ndarray
    mask: np.ndarray
    label_enabled: bool = True


@dataclass(slots=True)
class SourceSection:
    hu: np.ndarray
    preprocessed_hu: np.ndarray
    forward_max_mm: float
    shaft_half_width_mm: float
    width: int
    height: int


@dataclass(slots=True)
class LocalPoseOptimizationResult:
    device_pose: DevicePose
    branch_shift_mm: float
    roll_offset_deg: float
    axis_sign_override: str | None
    objective: tuple[float | int, ...]
    metrics: dict[str, object]


def _resolve_pose(report, *, preset_id: str, approach: str | None):
    matches = [pose for pose in report.poses if pose.preset_id == preset_id]
    if not matches:
        raise ValueError(f"Preset {preset_id!r} is not defined in the manifest.")

    if approach is None:
        if len(matches) != 1:
            available = ", ".join(sorted(pose.contact_approach for pose in matches))
            raise ValueError(f"Preset {preset_id!r} has multiple approaches. Choose one of: {available}")
        return matches[0]

    for pose in matches:
        if pose.contact_approach == approach:
            return pose

    available = ", ".join(sorted(pose.contact_approach for pose in matches))
    raise ValueError(f"Preset {preset_id!r} does not have approach {approach!r}. Available: {available}")


def _overlay_summary(config: OverlayConfig) -> list[str]:
    enabled: list[str] = []
    if config.airway_lumen_enabled:
        enabled.append("airway_lumen")
    if config.airway_wall_enabled:
        enabled.append("airway_wall")
    if config.station_enabled:
        enabled.append("station")
    enabled.extend(config.vessel_names)
    if config.target_enabled:
        enabled.append("target")
    if config.contact_enabled:
        enabled.append("contact")
    return enabled


def _resolve_overlay_config(
    manifest: CaseManifest,
    *,
    mode: str | None,
    airway_overlay: bool | None,
    airway_lumen_overlay: bool | None,
    airway_wall_overlay: bool | None,
    target_overlay: bool | None,
    contact_overlay: bool | None,
    station_overlay: bool | None,
    vessel_overlay_names: list[str] | None,
    diagnostic_panel: bool,
    virtual_ebus: bool,
    simulated_ebus: bool,
    show_legend: bool | None,
    label_overlays: bool | None,
    show_frustum: bool | None,
    min_contour_area_px: float,
    min_contour_length_px: float,
    single_vessel_name: str | None,
    preset_default_vessel_names: list[str] | None = None,
) -> OverlayConfig:
    resolved_mode = DEFAULT_RENDER_MODE if mode is None else mode
    if resolved_mode not in {"clean", "debug"}:
        raise ValueError(f"Unsupported render mode: {resolved_mode!r}")

    if resolved_mode == "clean":
        default_lumen = False
        default_wall = False
        default_station = False
        default_target = False
        default_contact = False
        default_vessels: list[str] = []
        default_legend = False
        default_labels = False
        default_frustum = False
    else:
        default_lumen = True
        default_wall = True
        default_station = True
        default_target = True
        default_contact = True
        default_vessels = list(preset_default_vessel_names or [])
        default_legend = True
        default_labels = True
        default_frustum = True

    if airway_overlay is not None:
        default_lumen = airway_overlay
        default_wall = airway_overlay

    resolved_vessels = list(default_vessels if vessel_overlay_names is None else vessel_overlay_names)
    if single_vessel_name is not None:
        resolved_vessels = [single_vessel_name]
    available_vessels = set(manifest.overlay_masks.keys())
    unknown_vessels = [name for name in resolved_vessels if name not in available_vessels]
    if unknown_vessels:
        available = ", ".join(sorted(available_vessels))
        missing = ", ".join(unknown_vessels)
        raise ValueError(f"Unknown vessel overlay(s): {missing}. Available overlay masks: {available}")

    return OverlayConfig(
        mode=resolved_mode,
        airway_lumen_enabled=(default_lumen if airway_lumen_overlay is None else airway_lumen_overlay),
        airway_wall_enabled=(default_wall if airway_wall_overlay is None else airway_wall_overlay),
        station_enabled=(default_station if station_overlay is None else station_overlay),
        target_enabled=(default_target if target_overlay is None else target_overlay),
        contact_enabled=(default_contact if contact_overlay is None else contact_overlay),
        vessel_names=resolved_vessels,
        diagnostic_panel_enabled=diagnostic_panel,
        virtual_ebus_enabled=virtual_ebus,
        simulated_ebus_enabled=simulated_ebus,
        show_legend=(default_legend if show_legend is None else show_legend),
        label_overlays=(default_labels if label_overlays is None else label_overlays),
        show_frustum=(default_frustum if show_frustum is None else show_frustum),
        min_contour_area_px=float(min_contour_area_px),
        min_contour_length_px=float(min_contour_length_px),
        single_vessel_name=single_vessel_name,
    )


def _resolve_slice_thickness_mm(mode: str, slice_thickness_mm: float | None) -> float:
    if slice_thickness_mm is not None:
        return float(slice_thickness_mm)
    return DEBUG_SLICE_THICKNESS_MM if mode == "debug" else CLEAN_SLICE_THICKNESS_MM


def _load_optional_polydata(path: Path | None) -> PolyData | None:
    if path is None or not path.exists():
        return None
    return load_vtp_polydata(path)


def _resolve_cutaway_config(
    *,
    cutaway_mode: str | None,
    cutaway_side: str | None,
    cutaway_depth_mm: float | None,
    cutaway_origin: str | None,
    show_full_airway: bool | None,
    default_side: str | None = None,
) -> CutawayConfig:
    resolved_side = cutaway_side if cutaway_side is not None else default_side
    return CutawayConfig(
        mode=("lateral" if cutaway_mode is None else cutaway_mode),
        side=("auto" if resolved_side is None else resolved_side),
        depth_mm=(0.0 if cutaway_depth_mm is None else float(cutaway_depth_mm)),
        origin_mode=("contact" if cutaway_origin is None else cutaway_origin),
        show_full_airway=(False if show_full_airway is None else bool(show_full_airway)),
    )


def _resolve_preset_manifest(manifest: CaseManifest, preset_id: str):
    for preset in manifest.presets:
        if preset.id == preset_id:
            return preset
    raise ValueError(f"Preset {preset_id!r} is not defined in the manifest.")


def build_render_context(manifest_path: str | Path, *, roll_deg: float | None = None) -> RenderContext:
    manifest = load_case_manifest(manifest_path)
    return RenderContext(
        manifest=manifest,
        pose_report=generate_pose_report(manifest.manifest_path, roll_deg=roll_deg),
        ct_volume=load_nifti(manifest.ct_image, kind="ct", load_data=True),
        airway_lumen_volume=load_nifti(manifest.airway_lumen_mask, kind="mask", load_data=True),
        airway_solid_volume=load_nifti(manifest.airway_solid_mask, kind="mask", load_data=True),
        airway_display_mesh=_load_optional_polydata(
            manifest.airway_cutaway_display_mesh if manifest.airway_cutaway_display_mesh is not None else manifest.airway_display_mesh
        ),
        airway_geometry_mesh=_load_optional_polydata(manifest.airway_raw_mesh),
        mask_cache={},
        main_graph=CenterlineGraph.from_vtp(str(manifest.centerline_main), name="main"),
        network_graph=CenterlineGraph.from_vtp(str(manifest.centerline_network), name="network"),
    )


def _candidate_branch_keys_for_hint(context: RenderContext, branch_hint: str | None) -> list[tuple[str, int]]:
    hint_spec = _parse_branch_hint(branch_hint)
    if hint_spec is None:
        return []

    keys: list[tuple[str, int]] = []
    for line_index in sorted(hint_spec.network_lines):
        if int(line_index) in context.network_graph.polylines_by_index:
            keys.append(("network", int(line_index)))
    for line_index in sorted(hint_spec.main_lines):
        if int(line_index) in context.main_graph.polylines_by_index:
            keys.append(("main", int(line_index)))
    for line_index in sorted(hint_spec.any_lines):
        if int(line_index) in context.network_graph.polylines_by_index:
            keys.append(("network", int(line_index)))
        elif int(line_index) in context.main_graph.polylines_by_index:
            keys.append(("main", int(line_index)))
    return keys


def _graph_for_key(context: RenderContext, graph_name: str) -> CenterlineGraph:
    return context.network_graph if graph_name == "network" else context.main_graph


def _derive_local_candidate_depth_axis(
    *,
    contact_seed_world: np.ndarray,
    target_world: np.ndarray,
    shaft_axis_world: np.ndarray,
    fallback_probe_axis_world: np.ndarray,
) -> np.ndarray:
    projected_target = _project_perpendicular(target_world - contact_seed_world, shaft_axis_world)
    candidate_depth = _normalize(projected_target)
    if candidate_depth is not None:
        return candidate_depth
    fallback = _normalize(_project_perpendicular(fallback_probe_axis_world, shaft_axis_world))
    if fallback is not None:
        return fallback
    return np.asarray(fallback_probe_axis_world, dtype=np.float64)


def _resolve_branch_shift_seed(
    context: RenderContext,
    *,
    pose,
    branch_hint: str,
    branch_shift_mm: float,
    fallback_contact_world: np.ndarray,
    fallback_probe_axis_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    branch_keys = _candidate_branch_keys_for_hint(context, branch_hint)
    if not branch_keys:
        return None

    target_world = np.asarray(pose.target_world, dtype=np.float64)
    for graph_name, line_index in branch_keys:
        graph = _graph_for_key(context, graph_name)
        polyline = graph.polylines_by_index.get(int(line_index))
        if polyline is None:
            continue
        base_projection = graph.nearest_point(fallback_contact_world)
        if base_projection is not None and int(base_projection.line_index) == int(line_index):
            base_arclength_mm = float(base_projection.line_arclength_mm)
        else:
            markup_projection = graph.nearest_point(np.asarray(pose.contact_world, dtype=np.float64))
            if markup_projection is not None and int(markup_projection.line_index) == int(line_index):
                base_arclength_mm = float(markup_projection.line_arclength_mm)
            else:
                base_arclength_mm = float(polyline.total_length_mm / 2.0)
        candidate_s = float(np.clip(base_arclength_mm + float(branch_shift_mm), 0.0, polyline.total_length_mm))
        contact_seed_world = polyline.point_at_arc_length(candidate_s)
        shaft_axis_world = graph.estimate_tangent(line_index=int(line_index), line_arclength_mm=candidate_s)
        shaft_axis_world = _normalize(shaft_axis_world) if shaft_axis_world is not None else None
        if shaft_axis_world is None:
            continue
        depth_axis_world = _derive_local_candidate_depth_axis(
            contact_seed_world=contact_seed_world,
            target_world=target_world,
            shaft_axis_world=shaft_axis_world,
            fallback_probe_axis_world=fallback_probe_axis_world,
        )
        return contact_seed_world, shaft_axis_world, depth_axis_world
    return None


def _get_mask_volume(context: RenderContext, mask_path: Path) -> VolumeData:
    key = str(mask_path.resolve())
    cached = context.mask_cache.get(key)
    if cached is not None:
        return cached
    volume = load_nifti(mask_path, kind="mask", load_data=True)
    context.mask_cache[key] = volume
    return volume


def _build_optimization_cache_key(
    *,
    state_context: RenderContext,
    pose,
    device: str,
    configured_branch_hint: str | None,
    configured_branch_shift_mm: float | None,
    resolved_width: int,
    resolved_height: int,
    resolved_source_oblique_size_mm: float,
    resolved_max_depth_mm: float,
    resolved_sector_angle_deg: float,
    resolved_slice_thickness_mm: float,
    preset_roll_offset_deg: float,
    configured_axis_sign_override: str | None,
) -> tuple[object, ...] | None:
    if configured_branch_hint is None or configured_branch_shift_mm is not None:
        return None
    return (
        str(state_context.manifest.manifest_path),
        pose.preset_id,
        pose.contact_approach,
        device,
        configured_branch_hint,
        resolved_width,
        resolved_height,
        round(resolved_source_oblique_size_mm, 4),
        round(resolved_max_depth_mm, 4),
        round(resolved_sector_angle_deg, 4),
        round(resolved_slice_thickness_mm, 4),
        round(preset_roll_offset_deg, 4),
        configured_axis_sign_override,
    )


def _refresh_prepared_render_state(
    state: PreparedRenderState,
    *,
    device_pose: DevicePose,
    preset_roll_offset_deg: float,
    configured_axis_sign_override: str | None,
    optimization_result: LocalPoseOptimizationResult | None,
) -> PreparedRenderState:
    state.device_pose = device_pose
    state.preset_roll_offset_deg = float(preset_roll_offset_deg)
    state.configured_axis_sign_override = configured_axis_sign_override
    state.optimization_result = optimization_result
    state.resolved_roll_deg = float(state.pose.roll_deg + state.preset_roll_offset_deg)
    state.resolved_reference_fov_mm = float(
        (
            device_pose.device_model.reference_fov_mm
            if state.preset_overrides is None or state.preset_overrides.reference_fov_mm is None
            else state.preset_overrides.reference_fov_mm
        )
        if state.request.reference_fov_mm is None
        else state.request.reference_fov_mm
    )
    state.contact_world = np.asarray(device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    state.original_contact_world = np.asarray(device_pose.contact_refinement.original_contact_world, dtype=np.float64)
    state.target_world = np.asarray(device_pose.target_world, dtype=np.float64)
    state.probe_axis = np.asarray(device_pose.probe_axis_world, dtype=np.float64)
    state.shaft_axis = np.asarray(device_pose.shaft_axis_world, dtype=np.float64)
    state.lateral_axis = np.asarray(device_pose.lateral_axis_world, dtype=np.float64)
    return state


def _apply_optional_optimization(
    state: PreparedRenderState,
    *,
    optimize_pose_fn: Callable[..., LocalPoseOptimizationResult | None] | None,
) -> PreparedRenderState:
    if optimize_pose_fn is None or state.optimization_cache_key is None or state.configured_branch_hint is None:
        return state

    optimization_result = _LOCAL_POSE_OPTIMIZATION_CACHE.get(state.optimization_cache_key)
    if optimization_result is None:
        optimization_result = optimize_pose_fn(
            state.context,
            pose=state.pose,
            preset_manifest=state.preset_manifest,
            device=state.request.device,
            branch_hint=state.configured_branch_hint,
            base_device_pose=state.device_pose,
            base_roll_offset_deg=state.preset_roll_offset_deg,
            base_axis_sign_override=state.configured_axis_sign_override,
            width=state.resolved_width,
            height=state.resolved_height,
            source_oblique_size_mm=state.resolved_source_oblique_size_mm,
            max_depth_mm=state.resolved_max_depth_mm,
            sector_angle_deg=state.resolved_sector_angle_deg,
            slice_thickness_mm=state.resolved_slice_thickness_mm,
        )
        if optimization_result is not None:
            _LOCAL_POSE_OPTIMIZATION_CACHE[state.optimization_cache_key] = optimization_result

    if optimization_result is None:
        return state

    return _refresh_prepared_render_state(
        state,
        device_pose=optimization_result.device_pose,
        preset_roll_offset_deg=optimization_result.roll_offset_deg,
        configured_axis_sign_override=optimization_result.axis_sign_override,
        optimization_result=optimization_result,
    )


def _prepare_render_state(
    request: RenderRequest,
    *,
    context: RenderContext | None,
    target_overlay: bool | None,
    contact_overlay: bool | None,
    diagnostic_panel: bool,
    virtual_ebus: bool,
    simulated_ebus: bool,
    show_frustum: bool | None,
    optimize_pose_fn: Callable[..., LocalPoseOptimizationResult | None] | None,
) -> PreparedRenderState:
    render_context = build_render_context(request.manifest_path, roll_deg=request.roll_deg) if context is None else context
    manifest = render_context.manifest
    defaults = manifest.render_defaults
    pose = _resolve_pose(render_context.pose_report, preset_id=request.preset_id, approach=request.approach)
    preset_manifest = _resolve_preset_manifest(manifest, pose.preset_id)
    preset_overrides = resolve_preset_overrides(preset_manifest, approach=pose.contact_approach)

    overlay_config = _resolve_overlay_config(
        manifest,
        mode=request.mode,
        airway_overlay=request.airway_overlay,
        airway_lumen_overlay=request.airway_lumen_overlay,
        airway_wall_overlay=request.airway_wall_overlay,
        target_overlay=target_overlay,
        contact_overlay=contact_overlay,
        station_overlay=request.station_overlay,
        vessel_overlay_names=request.vessel_overlay_names,
        diagnostic_panel=diagnostic_panel,
        virtual_ebus=virtual_ebus,
        simulated_ebus=simulated_ebus,
        show_legend=request.show_legend,
        label_overlays=request.label_overlays,
        show_frustum=show_frustum,
        min_contour_area_px=request.min_contour_area_px,
        min_contour_length_px=request.min_contour_length_px,
        single_vessel_name=request.single_vessel,
        preset_default_vessel_names=(None if preset_overrides is None else preset_overrides.vessel_overlays),
    )
    cutaway_config = _resolve_cutaway_config(
        cutaway_mode=request.cutaway_mode,
        cutaway_side=request.cutaway_side,
        cutaway_depth_mm=request.cutaway_depth_mm,
        cutaway_origin=request.cutaway_origin,
        show_full_airway=request.show_full_airway,
        default_side=(None if preset_overrides is None else preset_overrides.cutaway_side),
    )
    if not overlay_config.virtual_ebus_enabled and not overlay_config.simulated_ebus_enabled:
        raise ValueError("At least one of virtual_ebus or simulated_ebus must be enabled.")

    resolved_width = int(defaults.get("image_size", [512, 512])[0] if request.width is None else request.width)
    resolved_height = int(defaults.get("image_size", [512, 512])[1] if request.height is None else request.height)
    preset_roll_offset_deg = 0.0 if preset_overrides is None or preset_overrides.roll_offset_deg is None else float(preset_overrides.roll_offset_deg)
    configured_axis_sign_override = None if preset_overrides is None else preset_overrides.axis_sign_override
    configured_branch_hint = None if preset_overrides is None else preset_overrides.branch_hint
    configured_branch_shift_mm = None if preset_overrides is None else preset_overrides.branch_shift_mm
    resolved_gain = float(defaults.get("gain", 1.0))
    resolved_attenuation = float(defaults.get("attenuation", 0.15))
    resolved_slice_thickness_mm = _resolve_slice_thickness_mm(overlay_config.mode, request.slice_thickness_mm)
    fallback_probe_axis_world = np.asarray(
        pose.depth_axis if pose.depth_axis is not None else pose.default_depth_axis,
        dtype=np.float64,
    )
    branch_shift_seed = (
        None
        if configured_branch_hint is None or configured_branch_shift_mm is None
        else _resolve_branch_shift_seed(
            render_context,
            pose=pose,
            branch_hint=configured_branch_hint,
            branch_shift_mm=float(configured_branch_shift_mm),
            fallback_contact_world=np.asarray(pose.contact_world, dtype=np.float64),
            fallback_probe_axis_world=fallback_probe_axis_world,
        )
    )
    device_pose = build_device_pose(
        pose,
        device_name=request.device,
        ct_volume=render_context.ct_volume,
        airway_lumen=render_context.airway_lumen_volume,
        airway_solid=render_context.airway_solid_volume,
        raw_airway_mesh=render_context.airway_geometry_mesh,
        main_graph=render_context.main_graph,
        network_graph=render_context.network_graph,
        refine_contact=request.refine_contact,
        roll_offset_deg=preset_roll_offset_deg,
        axis_sign_override=configured_axis_sign_override,
        branch_hint=configured_branch_hint,
        contact_seed_world=(None if branch_shift_seed is None else branch_shift_seed[0]),
        shaft_axis_override=(None if branch_shift_seed is None else branch_shift_seed[1]),
        depth_axis_override=(None if branch_shift_seed is None else branch_shift_seed[2]),
    )
    baseline_device_pose = device_pose

    resolved_sector_angle_deg = float(device_pose.device_model.sector_angle_deg if request.sector_angle_deg is None else request.sector_angle_deg)
    resolved_max_depth_mm = float(device_pose.device_model.displayed_range_mm if request.max_depth_mm is None else request.max_depth_mm)
    resolved_source_oblique_size_mm = float(device_pose.device_model.source_oblique_size_mm if request.source_oblique_size_mm is None else request.source_oblique_size_mm)

    state = PreparedRenderState(
        state_version=PREPARED_RENDER_STATE_VERSION,
        request=request,
        context=render_context,
        manifest=manifest,
        pose=pose,
        preset_manifest=preset_manifest,
        preset_overrides=preset_overrides,
        overlay_config=overlay_config,
        cutaway_config=cutaway_config,
        baseline_device_pose=baseline_device_pose,
        device_pose=device_pose,
        optimization_cache_key=_build_optimization_cache_key(
            state_context=render_context,
            pose=pose,
            device=request.device,
            configured_branch_hint=configured_branch_hint,
            configured_branch_shift_mm=configured_branch_shift_mm,
            resolved_width=resolved_width,
            resolved_height=resolved_height,
            resolved_source_oblique_size_mm=resolved_source_oblique_size_mm,
            resolved_max_depth_mm=resolved_max_depth_mm,
            resolved_sector_angle_deg=resolved_sector_angle_deg,
            resolved_slice_thickness_mm=resolved_slice_thickness_mm,
            preset_roll_offset_deg=preset_roll_offset_deg,
            configured_axis_sign_override=configured_axis_sign_override,
        ),
        optimization_result=None,
        resolved_width=resolved_width,
        resolved_height=resolved_height,
        resolved_sector_angle_deg=resolved_sector_angle_deg,
        resolved_max_depth_mm=resolved_max_depth_mm,
        resolved_source_oblique_size_mm=resolved_source_oblique_size_mm,
        resolved_reference_fov_mm=0.0,
        resolved_roll_deg=0.0,
        resolved_gain=resolved_gain,
        resolved_attenuation=resolved_attenuation,
        resolved_slice_thickness_mm=resolved_slice_thickness_mm,
        preset_roll_offset_deg=preset_roll_offset_deg,
        configured_axis_sign_override=configured_axis_sign_override,
        configured_branch_hint=configured_branch_hint,
        configured_branch_shift_mm=configured_branch_shift_mm,
        branch_shift_seed=branch_shift_seed,
        contact_world=np.asarray([], dtype=np.float64),
        original_contact_world=np.asarray([], dtype=np.float64),
        target_world=np.asarray([], dtype=np.float64),
        probe_axis=np.asarray([], dtype=np.float64),
        shaft_axis=np.asarray([], dtype=np.float64),
        lateral_axis=np.asarray([], dtype=np.float64),
    )
    _refresh_prepared_render_state(
        state,
        device_pose=device_pose,
        preset_roll_offset_deg=preset_roll_offset_deg,
        configured_axis_sign_override=configured_axis_sign_override,
        optimization_result=None,
    )
    return _apply_optional_optimization(state, optimize_pose_fn=optimize_pose_fn)


def prepare_localizer_render_state(
    request: RenderRequest,
    *,
    context: RenderContext | None = None,
    optimize_pose_fn: Callable[..., LocalPoseOptimizationResult | None] | None = None,
) -> PreparedRenderState:
    return _prepare_render_state(
        request,
        context=context,
        target_overlay=request.target_overlay,
        contact_overlay=(request.show_contact if request.contact_overlay is None else request.contact_overlay),
        diagnostic_panel=request.diagnostic_panel,
        virtual_ebus=request.virtual_ebus,
        simulated_ebus=request.simulated_ebus,
        show_frustum=request.show_frustum,
        optimize_pose_fn=optimize_pose_fn,
    )


def prepare_physics_render_state(
    request: RenderRequest,
    *,
    context: RenderContext | None = None,
    optimize_pose_fn: Callable[..., LocalPoseOptimizationResult | None] | None = None,
) -> PreparedRenderState:
    if not request.virtual_ebus and not request.simulated_ebus:
        raise ValueError("At least one of virtual_ebus or simulated_ebus must be enabled.")
    return _prepare_render_state(
        request,
        context=context,
        target_overlay=(False if request.target_overlay is None else request.target_overlay),
        contact_overlay=(request.show_contact if request.contact_overlay is None else request.contact_overlay),
        diagnostic_panel=False,
        virtual_ebus=False,
        simulated_ebus=True,
        show_frustum=False,
        optimize_pose_fn=optimize_pose_fn,
    )
