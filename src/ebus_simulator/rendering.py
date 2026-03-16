from __future__ import annotations

from csv import DictWriter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ebus_simulator.centerline import CenterlineGraph
from ebus_simulator.cutaway import CutawayDisplay, build_display_cutaway
from ebus_simulator.device import DevicePose, build_device_pose
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.io.vtp import load_vtp_polydata
from ebus_simulator.manifest import load_case_manifest
from ebus_simulator.models import CaseManifest, PolyData, VolumeData
from ebus_simulator.poses import generate_pose_report


WINDOW_CENTER_HU = 40.0
WINDOW_WIDTH_HU = 300.0
DEFAULT_SLAB_SAMPLES = 5
CLEAN_SLICE_THICKNESS_MM = 4.0
DEBUG_SLICE_THICKNESS_MM = 1.5
DEFAULT_RENDER_MODE = "debug"
CONTEXT_AXIS_LENGTH_MM = 12.0
CONTEXT_CENTERLINE_WINDOW_MM = 28.0
DEFAULT_DEVICE_NAME = "bf_uc180f"
DEFAULT_REFERENCE_FOV_MM = 100.0
DEFAULT_SOURCE_OBLIQUE_SIZE_MM = 51.79
SOURCE_SECTION_PREPROCESS_BLEND = 0.30

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
class RenderMetadata:
    manifest_path: str
    case_id: str
    preset_id: str
    approach: str
    mode: str
    output_path: str
    metadata_path: str
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
    warnings: list[str]


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


def _points_to_voxel(points_lps: np.ndarray, inverse_affine_lps: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points_lps, np.ones((points_lps.shape[0], 1), dtype=np.float64)), axis=1)
    ijk = homogeneous @ inverse_affine_lps.T
    return ijk[:, :3]


def _sample_slab(
    data: np.ndarray,
    *,
    base_points_lps: np.ndarray,
    thickness_axis: np.ndarray,
    inverse_affine_lps: np.ndarray,
    sample_count: int,
    slab_thickness_mm: float,
    order: int,
    cval: float,
) -> np.ndarray:
    if base_points_lps.size == 0:
        return np.asarray([], dtype=np.float32)

    if sample_count <= 1 or slab_thickness_mm <= 0.0:
        offsets = np.asarray([0.0], dtype=np.float64)
    else:
        offsets = np.linspace(-slab_thickness_mm / 2.0, slab_thickness_mm / 2.0, sample_count, dtype=np.float64)

    stacked_points = base_points_lps[None, :, :] + offsets[:, None, None] * thickness_axis[None, None, :]
    voxel_points = _points_to_voxel(stacked_points.reshape((-1, 3)), inverse_affine_lps)
    sampled = ndimage.map_coordinates(
        np.asarray(data, dtype=np.float32),
        [voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]],
        order=order,
        mode="constant",
        cval=cval,
    )
    return sampled.reshape((offsets.shape[0], base_points_lps.shape[0])).mean(axis=0)


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


def _build_sector_grid(width: int, height: int, max_depth_mm: float, sector_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    tan_half = float(np.tan(np.deg2rad(sector_angle_deg / 2.0)))
    max_lateral_mm = max_depth_mm * tan_half
    depth_mm = (np.arange(height, dtype=np.float64) / max(1, height - 1)) * max_depth_mm
    lateral_mm = ((np.arange(width, dtype=np.float64) - (width // 2)) / max(1, width // 2)) * max_lateral_mm
    depth_grid = np.broadcast_to(depth_mm[:, None], (height, width))
    lateral_grid = np.broadcast_to(lateral_mm[None, :], (height, width))
    sector_mask = np.abs(lateral_grid) <= ((depth_grid * tan_half) + 1e-9)
    return depth_grid, lateral_grid, sector_mask, max_lateral_mm


def _window_ct(values_hu: np.ndarray, *, gain: float, attenuation: float, depths_mm: np.ndarray, max_depth_mm: float) -> np.ndarray:
    window_min = WINDOW_CENTER_HU - (WINDOW_WIDTH_HU / 2.0)
    normalized = np.clip((values_hu - window_min) / WINDOW_WIDTH_HU, 0.0, 1.0)
    attenuation_curve = np.exp(-attenuation * (depths_mm / max(max_depth_mm, 1e-6)))
    return np.clip(normalized * gain * attenuation_curve, 0.0, 1.0)


def _compute_contour(binary_mask: np.ndarray) -> np.ndarray:
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, border_value=0)
    contour = np.logical_and(binary_mask, np.logical_not(eroded))
    return ndimage.binary_dilation(contour, iterations=1)


def _apply_contour_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, color_rgb: np.ndarray) -> None:
    contour = _compute_contour(binary_mask)
    if not np.any(contour):
        return
    image_rgb[contour] = image_rgb[contour] * 0.12 + color_rgb[None, :] * 0.88


def _draw_cross_marker(image_rgb: np.ndarray, *, row: int, column: int, color_rgb: np.ndarray, radius: int = 4) -> None:
    height, width = image_rgb.shape[:2]
    for offset in range(-radius, radius + 1):
        current_row = row + offset
        current_column = column + offset
        opposite_column = column - offset
        if 0 <= current_row < height and 0 <= column < width:
            image_rgb[current_row, column] = image_rgb[current_row, column] * 0.10 + color_rgb * 0.90
        if 0 <= row < height and 0 <= current_column < width:
            image_rgb[row, current_column] = image_rgb[row, current_column] * 0.10 + color_rgb * 0.90
        if 0 <= current_row < height and 0 <= current_column < width:
            image_rgb[current_row, current_column] = image_rgb[current_row, current_column] * 0.25 + color_rgb * 0.75
        if 0 <= current_row < height and 0 <= opposite_column < width:
            image_rgb[current_row, opposite_column] = image_rgb[current_row, opposite_column] * 0.25 + color_rgb * 0.75


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
        default_vessels = []
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
) -> CutawayConfig:
    return CutawayConfig(
        mode=("lateral" if cutaway_mode is None else cutaway_mode),
        side=("auto" if cutaway_side is None else cutaway_side),
        depth_mm=(0.0 if cutaway_depth_mm is None else float(cutaway_depth_mm)),
        origin_mode=("contact" if cutaway_origin is None else cutaway_origin),
        show_full_airway=(False if show_full_airway is None else bool(show_full_airway)),
    )


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


def _get_mask_volume(context: RenderContext, mask_path: Path) -> VolumeData:
    key = str(mask_path.resolve())
    cached = context.mask_cache.get(key)
    if cached is not None:
        return cached
    volume = load_nifti(mask_path, kind="mask", load_data=True)
    context.mask_cache[key] = volume
    return volume


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


def _sample_plane(
    context: RenderContext,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    thickness_axis: np.ndarray,
    center_world: np.ndarray,
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    width: int,
    height: int,
    slice_thickness_mm: float,
    order: int,
    cval: float,
    data: np.ndarray,
    inverse_affine_lps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.linspace(x_min_mm, x_max_mm, width, dtype=np.float64)
    y_coords = np.linspace(y_max_mm, y_min_mm, height, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    plane_points = (
        center_world[None, None, :]
        + x_grid[:, :, None] * x_axis[None, None, :]
        + y_grid[:, :, None] * y_axis[None, None, :]
    )
    sampled = _sample_slab(
        np.asarray(data, dtype=np.float32),
        base_points_lps=plane_points.reshape((-1, 3)),
        thickness_axis=thickness_axis,
        inverse_affine_lps=inverse_affine_lps,
        sample_count=DEFAULT_SLAB_SAMPLES,
        slab_thickness_mm=slice_thickness_mm,
        order=order,
        cval=cval,
    )
    return sampled.reshape((height, width)), x_coords, y_coords, plane_points


def _plane_point_to_pixel(x_mm: float, y_mm: float, *, x_min_mm: float, x_max_mm: float, y_min_mm: float, y_max_mm: float, width: int, height: int) -> tuple[int, int]:
    column = int(round((x_mm - x_min_mm) / max(x_max_mm - x_min_mm, 1e-6) * (width - 1)))
    row = int(round((y_max_mm - y_mm) / max(y_max_mm - y_min_mm, 1e-6) * (height - 1)))
    return row, column


def _to_uint8(image_rgb: np.ndarray) -> np.ndarray:
    return np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _add_panel_label(image_rgb: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(_to_uint8(image_rgb), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 140, 20), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


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


def _sample_contact_plane(
    *,
    data: np.ndarray,
    inverse_affine_lps: np.ndarray,
    origin_world: np.ndarray,
    depth_axis: np.ndarray,
    lateral_axis: np.ndarray,
    thickness_axis: np.ndarray,
    depth_max_mm: float,
    lateral_half_width_mm: float,
    width: int,
    height: int,
    slice_thickness_mm: float,
    order: int,
    cval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_coords = np.linspace(0.0, depth_max_mm, height, dtype=np.float64)
    lateral_coords = np.linspace(-lateral_half_width_mm, lateral_half_width_mm, width, dtype=np.float64)
    depth_grid = np.broadcast_to(depth_coords[:, None], (height, width))
    lateral_grid = np.broadcast_to(lateral_coords[None, :], (height, width))
    plane_points = (
        origin_world[None, None, :]
        + depth_grid[:, :, None] * depth_axis[None, None, :]
        + lateral_grid[:, :, None] * lateral_axis[None, None, :]
    )
    sampled = _sample_slab(
        np.asarray(data, dtype=np.float32),
        base_points_lps=plane_points.reshape((-1, 3)),
        thickness_axis=thickness_axis,
        inverse_affine_lps=inverse_affine_lps,
        sample_count=DEFAULT_SLAB_SAMPLES,
        slab_thickness_mm=slice_thickness_mm,
        order=order,
        cval=cval,
    )
    return sampled.reshape((height, width)), depth_grid, lateral_grid, plane_points


def _preprocess_source_section(source_hu: np.ndarray) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(source_hu.astype(np.float32), sigma=0.9)
    detail = source_hu.astype(np.float32) - smoothed
    return smoothed + (SOURCE_SECTION_PREPROCESS_BLEND * detail)


def _fan_plane_coordinates(
    depth_grid_mm: np.ndarray,
    display_lateral_grid_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(display_lateral_grid_mm, np.maximum(depth_grid_mm, 1e-9))
    forward_mm = depth_grid_mm * np.cos(phi)
    shaft_mm = depth_grid_mm * np.sin(phi)
    return forward_mm, shaft_mm


def _map_plane_to_fan(
    source_plane: np.ndarray,
    *,
    source_forward_max_mm: float,
    source_shaft_half_width_mm: float,
    depth_grid_mm: np.ndarray,
    display_lateral_grid_mm: np.ndarray,
    sector_mask: np.ndarray,
    order: int,
    cval: float,
) -> np.ndarray:
    mapped = np.full(depth_grid_mm.shape, cval, dtype=np.float32)
    if not np.any(sector_mask):
        return mapped

    forward_mm, shaft_mm = _fan_plane_coordinates(depth_grid_mm, display_lateral_grid_mm)
    source_rows = (forward_mm[sector_mask] / max(source_forward_max_mm, 1e-6)) * max(1, source_plane.shape[0] - 1)
    source_cols = (
        (shaft_mm[sector_mask] + source_shaft_half_width_mm)
        / max(source_shaft_half_width_mm * 2.0, 1e-6)
        * max(1, source_plane.shape[1] - 1)
    )
    mapped_values = ndimage.map_coordinates(
        np.asarray(source_plane, dtype=np.float32),
        [source_rows, source_cols],
        order=order,
        mode="constant",
        cval=cval,
    )
    mapped[sector_mask] = mapped_values
    return mapped


def _filter_mask_components(binary_mask: np.ndarray, *, min_area_px: float, min_length_px: float) -> np.ndarray:
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    labeled, component_count = ndimage.label(binary_mask)
    filtered = np.zeros_like(binary_mask, dtype=bool)
    for component_index in range(1, component_count + 1):
        component = labeled == component_index
        area = float(np.count_nonzero(component))
        contour_length = float(np.count_nonzero(_compute_contour(component)))
        if area < min_area_px or contour_length < min_length_px:
            continue
        filtered |= component
    return filtered


def _fan_target_row_col(
    *,
    contact_world: np.ndarray,
    target_world: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    max_depth_mm: float,
    sector_angle_deg: float,
    width: int,
    height: int,
    max_lateral_mm: float,
) -> tuple[int, int] | None:
    target_offset = target_world - contact_world
    target_forward_mm = float(np.dot(target_offset, probe_axis))
    target_shaft_mm = float(np.dot(target_offset, shaft_axis))
    target_depth_mm = float(np.linalg.norm([target_forward_mm, target_shaft_mm]))
    if target_forward_mm <= 0.0:
        return None
    target_phi = float(np.arctan2(target_shaft_mm, target_forward_mm))
    target_lateral_mm = float(target_depth_mm * np.tan(target_phi))
    fan_half_tan = float(np.tan(np.deg2rad(sector_angle_deg / 2.0)))
    if not (0.0 <= target_depth_mm <= max_depth_mm):
        return None
    if abs(target_lateral_mm) > ((target_depth_mm * fan_half_tan) + 1e-9):
        return None
    row = int(round((target_depth_mm / max(max_depth_mm, 1e-6)) * (height - 1)))
    column = int(round((target_lateral_mm / max(max_lateral_mm, 1e-6)) * max(1, width // 2) + (width // 2)))
    return row, column


def _color_tuple(color_rgb: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * 255.0, 0.0, 255.0)) for channel in color_rgb.tolist())


def _annotate_legend_and_labels(
    image_rgb: np.ndarray,
    *,
    visible_layers: list[OverlayLayer],
    show_legend: bool,
    label_overlays: bool,
    legend_entries: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    image = Image.fromarray(_to_uint8(image_rgb), mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if label_overlays:
        label_layers = [layer for layer in visible_layers if layer.label_enabled and np.any(layer.mask)]
        for index, layer in enumerate(label_layers):
            points = np.argwhere(_compute_contour(layer.mask))
            if points.size == 0:
                continue
            center_row, center_col = points.mean(axis=0)
            label_on_left = bool(center_col > (width * 0.55))
            label_x = 10 if label_on_left else max(10, width - 156)
            label_y = 26 + (index * 18)
            text_color = _color_tuple(layer.color_rgb)
            draw.line(
                (
                    float(center_col),
                    float(center_row),
                    float(label_x + (120 if label_on_left else 0)),
                    float(label_y + 8),
                ),
                fill=text_color,
                width=2,
            )
            draw.rectangle((label_x, label_y, label_x + 146, label_y + 16), fill=(0, 0, 0))
            draw.text((label_x + 4, label_y + 2), layer.label, fill=text_color)

    if show_legend and legend_entries:
        legend_height = 8 + (18 * len(legend_entries))
        legend_width = 186
        legend_x = 10
        legend_y = max(10, height - legend_height - 10)
        draw.rectangle((legend_x, legend_y, legend_x + legend_width, legend_y + legend_height), fill=(0, 0, 0))
        for index, (label, color_rgb) in enumerate(legend_entries):
            top = legend_y + 4 + (index * 18)
            text_color = _color_tuple(color_rgb)
            draw.rectangle((legend_x + 6, top + 4, legend_x + 16, top + 14), fill=text_color)
            draw.text((legend_x + 22, top + 1), label, fill=text_color)

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


def render_preset(
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
    context: RenderContext | None = None,
) -> RenderedPreset:
    render_context = build_render_context(manifest_path, roll_deg=roll_deg) if context is None else context
    manifest = render_context.manifest
    defaults = manifest.render_defaults
    overlay_config = _resolve_overlay_config(
        manifest,
        mode=mode,
        airway_overlay=airway_overlay,
        airway_lumen_overlay=airway_lumen_overlay,
        airway_wall_overlay=airway_wall_overlay,
        target_overlay=target_overlay,
        contact_overlay=(show_contact if contact_overlay is None else contact_overlay),
        station_overlay=station_overlay,
        vessel_overlay_names=vessel_overlay_names,
        diagnostic_panel=diagnostic_panel,
        virtual_ebus=virtual_ebus,
        simulated_ebus=simulated_ebus,
        show_legend=show_legend,
        label_overlays=label_overlays,
        show_frustum=show_frustum,
        min_contour_area_px=min_contour_area_px,
        min_contour_length_px=min_contour_length_px,
        single_vessel_name=single_vessel,
    )
    cutaway_config = _resolve_cutaway_config(
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
    )
    if not overlay_config.virtual_ebus_enabled and not overlay_config.simulated_ebus_enabled:
        raise ValueError("At least one of virtual_ebus or simulated_ebus must be enabled.")

    resolved_width = int(defaults.get("image_size", [512, 512])[0] if width is None else width)
    resolved_height = int(defaults.get("image_size", [512, 512])[1] if height is None else height)
    resolved_roll_deg = float(defaults.get("roll_deg", 0.0) if roll_deg is None else roll_deg)
    resolved_gain = float(defaults.get("gain", 1.0))
    resolved_attenuation = float(defaults.get("attenuation", 0.15))
    resolved_slice_thickness_mm = _resolve_slice_thickness_mm(overlay_config.mode, slice_thickness_mm)

    pose = _resolve_pose(render_context.pose_report, preset_id=preset_id, approach=approach)
    preset_manifest = next(preset for preset in manifest.presets if preset.id == pose.preset_id)
    device_pose = build_device_pose(
        pose,
        device_name=device,
        ct_volume=render_context.ct_volume,
        airway_lumen=render_context.airway_lumen_volume,
        airway_solid=render_context.airway_solid_volume,
        raw_airway_mesh=render_context.airway_geometry_mesh,
        main_graph=render_context.main_graph,
        network_graph=render_context.network_graph,
        refine_contact=refine_contact,
    )

    resolved_sector_angle_deg = float(device_pose.device_model.sector_angle_deg if sector_angle_deg is None else sector_angle_deg)
    resolved_max_depth_mm = float(device_pose.device_model.displayed_range_mm if max_depth_mm is None else max_depth_mm)
    resolved_source_oblique_size_mm = float(device_pose.device_model.source_oblique_size_mm if source_oblique_size_mm is None else source_oblique_size_mm)
    resolved_reference_fov_mm = float(device_pose.device_model.reference_fov_mm if reference_fov_mm is None else reference_fov_mm)

    contact_world = np.asarray(device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    original_contact_world = np.asarray(device_pose.contact_refinement.original_contact_world, dtype=np.float64)
    target_world = np.asarray(device_pose.target_world, dtype=np.float64)
    shaft_axis = np.asarray(device_pose.shaft_axis_world, dtype=np.float64)
    probe_axis = np.asarray(device_pose.probe_axis_world, dtype=np.float64)
    lateral_axis = np.asarray(device_pose.lateral_axis_world, dtype=np.float64)
    station_local = _extract_local_mask_points(
        _get_mask_volume(render_context, preset_manifest.station_mask),
        center_world=contact_world,
        lateral_axis=lateral_axis,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        radius_mm=max(resolved_max_depth_mm, 32.0),
        max_points=1200,
    )
    station_visibility_points_world = (
        None
        if station_local.size == 0
        else (
            contact_world[None, :]
            + station_local[:, [0]] * lateral_axis[None, :]
            + station_local[:, [1]] * probe_axis[None, :]
            + station_local[:, [2]] * shaft_axis[None, :]
        )
    )
    if render_context.airway_display_mesh is None:
        raise ValueError("Smoothed airway display mesh is unavailable at meshes/airway_endoluminal_surface_smoothed.vtp.")
    cutaway_display = build_display_cutaway(
        render_context.airway_display_mesh,
        mesh_source="smoothed",
        station=pose.station,
        approach=pose.contact_approach,
        mode=cutaway_config.mode,
        requested_side=cutaway_config.side,
        origin_mode=cutaway_config.origin_mode,
        depth_mm=cutaway_config.depth_mm,
        show_full_airway=cutaway_config.show_full_airway,
        contact_world=contact_world,
        target_world=target_world,
        lateral_axis_world=lateral_axis,
        probe_axis_world=probe_axis,
        shaft_axis_world=shaft_axis,
        probe_origin_world=np.asarray(device_pose.probe_origin_world, dtype=np.float64),
        custom_origin_world=cutaway_custom_origin_world,
        station_visibility_points_world=station_visibility_points_world,
    )

    depth_grid, lateral_grid, sector_mask, max_lateral_mm = _build_sector_grid(
        resolved_width,
        resolved_height,
        resolved_max_depth_mm,
        resolved_sector_angle_deg,
    )

    source_hu, _, _, _ = _sample_contact_plane(
        data=np.asarray(render_context.ct_volume.data, dtype=np.float32),
        inverse_affine_lps=render_context.ct_volume.inverse_affine_lps,
        origin_world=contact_world,
        depth_axis=probe_axis,
        lateral_axis=shaft_axis,
        thickness_axis=lateral_axis,
        depth_max_mm=resolved_source_oblique_size_mm,
        lateral_half_width_mm=(resolved_source_oblique_size_mm / 2.0),
        width=resolved_width,
        height=resolved_height,
        slice_thickness_mm=resolved_slice_thickness_mm,
        order=1,
        cval=-1000.0,
    )
    source_section = SourceSection(
        hu=source_hu,
        preprocessed_hu=_preprocess_source_section(source_hu),
        forward_max_mm=resolved_source_oblique_size_mm,
        shaft_half_width_mm=(resolved_source_oblique_size_mm / 2.0),
        width=resolved_width,
        height=resolved_height,
    )

    fan_hu = _map_plane_to_fan(
        source_section.hu,
        source_forward_max_mm=source_section.forward_max_mm,
        source_shaft_half_width_mm=source_section.shaft_half_width_mm,
        depth_grid_mm=depth_grid,
        display_lateral_grid_mm=lateral_grid,
        sector_mask=sector_mask,
        order=1,
        cval=-1000.0,
    )
    fan_preprocessed_hu = _map_plane_to_fan(
        source_section.preprocessed_hu,
        source_forward_max_mm=source_section.forward_max_mm,
        source_shaft_half_width_mm=source_section.shaft_half_width_mm,
        depth_grid_mm=depth_grid,
        display_lateral_grid_mm=lateral_grid,
        sector_mask=sector_mask,
        order=1,
        cval=-1000.0,
    )
    base_windowed = _window_ct(
        fan_hu[sector_mask],
        gain=resolved_gain,
        attenuation=resolved_attenuation,
        depths_mm=depth_grid[sector_mask],
        max_depth_mm=resolved_max_depth_mm,
    )
    virtual_rgb = np.zeros((resolved_height, resolved_width, 3), dtype=np.float32)
    virtual_rgb[sector_mask] = np.repeat(base_windowed[:, None], 3, axis=1)
    _apply_contour_overlay(virtual_rgb, sector_mask, FAN_BOUNDARY_COLOR)

    visible_layers: list[OverlayLayer] = []
    source_layer_masks: dict[str, np.ndarray] = {}

    def _sample_source_mask_to_fan(mask_path: Path) -> np.ndarray:
        mask_volume = _get_mask_volume(render_context, mask_path)
        source_mask, _, _, _ = _sample_contact_plane(
            data=np.asarray(mask_volume.data, dtype=np.float32),
            inverse_affine_lps=mask_volume.inverse_affine_lps,
            origin_world=contact_world,
            depth_axis=probe_axis,
            lateral_axis=shaft_axis,
            thickness_axis=lateral_axis,
            depth_max_mm=resolved_source_oblique_size_mm,
            lateral_half_width_mm=(resolved_source_oblique_size_mm / 2.0),
            width=resolved_width,
            height=resolved_height,
            slice_thickness_mm=resolved_slice_thickness_mm,
            order=0,
            cval=0.0,
        )
        fan_mask = _map_plane_to_fan(
            source_mask,
            source_forward_max_mm=source_section.forward_max_mm,
            source_shaft_half_width_mm=source_section.shaft_half_width_mm,
            depth_grid_mm=depth_grid,
            display_lateral_grid_mm=lateral_grid,
            sector_mask=sector_mask,
            order=0,
            cval=0.0,
        ) > 0.0
        return fan_mask

    def _add_source_mask_layer(mask_path: Path, label: str, color_rgb: np.ndarray, *, label_enabled: bool) -> None:
        fan_mask = _sample_source_mask_to_fan(mask_path)
        source_layer_masks[label.lower().replace(" ", "_")] = fan_mask
        filtered_mask = _filter_mask_components(
            np.logical_and(fan_mask, sector_mask),
            min_area_px=overlay_config.min_contour_area_px,
            min_length_px=overlay_config.min_contour_length_px,
        )
        if not np.any(filtered_mask):
            return
        _apply_contour_overlay(virtual_rgb, filtered_mask, color_rgb)
        visible_layers.append(OverlayLayer(key=label.lower().replace(" ", "_"), label=label, color_rgb=color_rgb, mask=filtered_mask, label_enabled=label_enabled))

    fan_lumen_mask = np.logical_and(_sample_source_mask_to_fan(render_context.manifest.airway_lumen_mask), sector_mask)

    if overlay_config.airway_lumen_enabled:
        _add_source_mask_layer(render_context.manifest.airway_lumen_mask, "Airway lumen", AIRWAY_LUMEN_COLOR, label_enabled=False)
    if overlay_config.airway_wall_enabled:
        _add_source_mask_layer(render_context.manifest.airway_solid_mask, "Airway wall", AIRWAY_WALL_COLOR, label_enabled=False)
    if overlay_config.station_enabled:
        _add_source_mask_layer(preset_manifest.station_mask, f"Station {preset_manifest.station.upper()} ROI", STATION_COLOR, label_enabled=True)
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        _add_source_mask_layer(
            render_context.manifest.overlay_masks[vessel_name],
            vessel_name.replace("_", " ").title(),
            VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)],
            label_enabled=True,
        )

    target_pixel = _fan_target_row_col(
        contact_world=contact_world,
        target_world=target_world,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        width=resolved_width,
        height=resolved_height,
        max_lateral_mm=max_lateral_mm,
    )
    if overlay_config.target_enabled and target_pixel is not None:
        _draw_cross_marker(virtual_rgb, row=target_pixel[0], column=target_pixel[1], color_rgb=TARGET_MARKER_COLOR, radius=4)
    if overlay_config.contact_enabled:
        _draw_cross_marker(virtual_rgb, row=0, column=resolved_width // 2, color_rgb=CONTACT_MARKER_COLOR, radius=4)

    legend_entries = [(layer.label, layer.color_rgb) for layer in visible_layers]
    if overlay_config.target_enabled and target_pixel is not None:
        legend_entries.append(("Target", TARGET_MARKER_COLOR))
    if overlay_config.contact_enabled:
        legend_entries.append(("Refined contact", CONTACT_MARKER_COLOR))
    virtual_view = _annotate_legend_and_labels(
        virtual_rgb,
        visible_layers=visible_layers,
        show_legend=overlay_config.show_legend,
        label_overlays=overlay_config.label_overlays,
        legend_entries=legend_entries,
    )

    simulated_depth_gradient = np.abs(np.gradient(fan_preprocessed_hu, axis=0))
    simulated_lateral_gradient = np.abs(np.gradient(fan_preprocessed_hu, axis=1))
    interface_strength = np.clip((0.70 * simulated_depth_gradient + 0.30 * simulated_lateral_gradient) / 220.0, 0.0, 1.0)
    simulated_windowed = np.zeros((resolved_height, resolved_width), dtype=np.float32)
    simulated_windowed[sector_mask] = _window_ct(
        fan_preprocessed_hu[sector_mask],
        gain=resolved_gain,
        attenuation=resolved_attenuation,
        depths_mm=depth_grid[sector_mask],
        max_depth_mm=resolved_max_depth_mm,
    )
    attenuation_curve = np.exp(-resolved_attenuation * (depth_grid / max(resolved_max_depth_mm, 1e-6)))
    simulated_gray = np.clip((0.58 * simulated_windowed) + (0.42 * interface_strength * attenuation_curve), 0.0, 1.0)
    air_dominant_mask = np.logical_and(fan_preprocessed_hu < -800.0, sector_mask)
    suppression_mask = np.logical_or(fan_lumen_mask, air_dominant_mask)
    if np.any(suppression_mask):
        preserved_signal = np.where(~suppression_mask, simulated_gray, 0.0)
        ray_signal_envelope = np.maximum.accumulate(preserved_signal, axis=0)
        ray_interface_envelope = np.maximum.accumulate(interface_strength, axis=0)
        nearest_tissue_fill = np.zeros_like(simulated_gray)
        valid_signal_mask = np.logical_and(sector_mask, ~suppression_mask)
        if np.any(valid_signal_mask):
            _, nearest_indices = ndimage.distance_transform_edt(
                np.where(sector_mask, suppression_mask, True),
                return_indices=True,
            )
            nearest_tissue_fill = simulated_gray[nearest_indices[0], nearest_indices[1]]
            nearest_tissue_fill[~sector_mask] = 0.0
        suppressed_fill = np.clip(
            (0.18 * attenuation_curve)
            + (0.34 * ray_interface_envelope)
            + (0.36 * ray_signal_envelope),
            0.08,
            0.54,
        )
        simulated_gray[suppression_mask] = np.clip(
            np.maximum(suppressed_fill[suppression_mask], 0.72 * nearest_tissue_fill[suppression_mask]),
            0.10,
            0.58,
        )
    simulated_rgb = np.zeros((resolved_height, resolved_width, 3), dtype=np.float32)
    simulated_rgb[sector_mask] = np.repeat(simulated_gray[sector_mask, None], 3, axis=1)
    _apply_contour_overlay(simulated_rgb, sector_mask, FAN_BOUNDARY_COLOR * 0.9)
    simulated_view = _to_uint8(simulated_rgb)

    localizer_x_min_mm = -10.0
    localizer_x_max_mm = max(20.0, resolved_reference_fov_mm - 10.0)
    localizer_y_min_mm = -(resolved_reference_fov_mm / 2.0)
    localizer_y_max_mm = resolved_reference_fov_mm / 2.0
    localizer_ct, _, _, _ = _sample_plane(
        render_context,
        x_axis=probe_axis,
        y_axis=shaft_axis,
        thickness_axis=lateral_axis,
        center_world=contact_world,
        x_min_mm=localizer_x_min_mm,
        x_max_mm=localizer_x_max_mm,
        y_min_mm=localizer_y_min_mm,
        y_max_mm=localizer_y_max_mm,
        width=resolved_width,
        height=resolved_height,
        slice_thickness_mm=resolved_slice_thickness_mm,
        order=1,
        cval=-1000.0,
        data=np.asarray(render_context.ct_volume.data, dtype=np.float32),
        inverse_affine_lps=render_context.ct_volume.inverse_affine_lps,
    )
    localizer_rgb = np.repeat(
        _window_ct(
            localizer_ct.reshape(-1),
            gain=1.0,
            attenuation=0.0,
            depths_mm=np.zeros(resolved_width * resolved_height, dtype=np.float32),
            max_depth_mm=max(resolved_reference_fov_mm, 1.0),
        ).reshape((resolved_height, resolved_width))[:, :, None],
        3,
        axis=2,
    )
    localizer_layers: list[OverlayLayer] = []

    def _add_localizer_mask(mask_path: Path, label: str, color_rgb: np.ndarray, *, label_enabled: bool) -> None:
        mask_volume = _get_mask_volume(render_context, mask_path)
        samples, _, _, _ = _sample_plane(
            render_context,
            x_axis=probe_axis,
            y_axis=shaft_axis,
            thickness_axis=lateral_axis,
            center_world=contact_world,
            x_min_mm=localizer_x_min_mm,
            x_max_mm=localizer_x_max_mm,
            y_min_mm=localizer_y_min_mm,
            y_max_mm=localizer_y_max_mm,
            width=resolved_width,
            height=resolved_height,
            slice_thickness_mm=resolved_slice_thickness_mm,
            order=0,
            cval=0.0,
            data=np.asarray(mask_volume.data, dtype=np.float32),
            inverse_affine_lps=mask_volume.inverse_affine_lps,
        )
        filtered_mask = _filter_mask_components(
            samples > 0.0,
            min_area_px=overlay_config.min_contour_area_px,
            min_length_px=overlay_config.min_contour_length_px,
        )
        if not np.any(filtered_mask):
            return
        _apply_contour_overlay(localizer_rgb, filtered_mask, color_rgb)
        localizer_layers.append(OverlayLayer(key=label.lower().replace(" ", "_"), label=label, color_rgb=color_rgb, mask=filtered_mask, label_enabled=label_enabled))

    if overlay_config.airway_lumen_enabled:
        _add_localizer_mask(render_context.manifest.airway_lumen_mask, "Airway lumen", AIRWAY_LUMEN_COLOR, label_enabled=False)
    if overlay_config.airway_wall_enabled:
        _add_localizer_mask(render_context.manifest.airway_solid_mask, "Airway wall", AIRWAY_WALL_COLOR, label_enabled=False)
    if overlay_config.station_enabled:
        _add_localizer_mask(preset_manifest.station_mask, f"Station {preset_manifest.station.upper()} ROI", STATION_COLOR, label_enabled=True)
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        _add_localizer_mask(
            render_context.manifest.overlay_masks[vessel_name],
            vessel_name.replace("_", " ").title(),
            VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)],
            label_enabled=True,
        )

    if overlay_config.contact_enabled:
        row, column = _plane_point_to_pixel(
            0.0,
            0.0,
            x_min_mm=localizer_x_min_mm,
            x_max_mm=localizer_x_max_mm,
            y_min_mm=localizer_y_min_mm,
            y_max_mm=localizer_y_max_mm,
            width=resolved_width,
            height=resolved_height,
        )
        _draw_cross_marker(localizer_rgb, row=row, column=column, color_rgb=CONTACT_MARKER_COLOR, radius=4)
    if overlay_config.target_enabled:
        target_offset = target_world - contact_world
        row, column = _plane_point_to_pixel(
            float(np.dot(target_offset, probe_axis)),
            float(np.dot(target_offset, shaft_axis)),
            x_min_mm=localizer_x_min_mm,
            x_max_mm=localizer_x_max_mm,
            y_min_mm=localizer_y_min_mm,
            y_max_mm=localizer_y_max_mm,
            width=resolved_width,
            height=resolved_height,
        )
        _draw_cross_marker(localizer_rgb, row=row, column=column, color_rgb=TARGET_MARKER_COLOR, radius=4)
    localizer_view = _annotate_legend_and_labels(
        localizer_rgb,
        visible_layers=localizer_layers,
        show_legend=False,
        label_overlays=overlay_config.label_overlays,
        legend_entries=[],
    )

    context_snapshot = _build_cp_context_snapshot(
        render_context,
        pose=pose,
        device_pose=device_pose,
        overlay_config=overlay_config,
        cutaway_display=cutaway_display,
        station_local=station_local,
        width=resolved_width,
        height=resolved_height,
        max_depth_mm=resolved_max_depth_mm,
    )

    if overlay_config.diagnostic_panel_enabled:
        output_image_uint8 = _compose_cp_diagnostic_panel(virtual_view, simulated_view, localizer_view, context_snapshot)
        panel_layout = ["virtual_ebus", "simulated_ebus", "wide_ct_localizer", "context_3d"]
    else:
        if overlay_config.virtual_ebus_enabled and (_overlay_summary(overlay_config) or overlay_config.mode == "debug"):
            output_image_uint8 = virtual_view
        else:
            output_image_uint8 = simulated_view
        panel_layout = []

    output_path = Path(output_path).expanduser().resolve()
    metadata_path = output_path.with_suffix(".json") if metadata_path is None else Path(metadata_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_image_uint8, mode="RGB").save(output_path)

    warnings = list(pose.warnings) + list(device_pose.contact_refinement.warnings) + list(cutaway_display.warnings)
    if overlay_config.single_vessel_name is not None and not any(layer.label == overlay_config.single_vessel_name.replace("_", " ").title() for layer in visible_layers):
        warnings.append(f"Single-vessel mode requested '{overlay_config.single_vessel_name}' but no contour intersected the displayed fan.")

    metadata = RenderMetadata(
        manifest_path=str(manifest.manifest_path),
        case_id=manifest.case_id,
        preset_id=pose.preset_id,
        approach=pose.contact_approach,
        mode=overlay_config.mode,
        output_path=str(output_path),
        metadata_path=str(metadata_path),
        image_size=[int(output_image_uint8.shape[1]), int(output_image_uint8.shape[0])],
        device_model=device_pose.device_model.name,
        device_label=device_pose.device_model.shaft_label,
        sector_angle_deg=resolved_sector_angle_deg,
        max_depth_mm=resolved_max_depth_mm,
        roll_deg=resolved_roll_deg,
        gain=resolved_gain,
        attenuation=resolved_attenuation,
        slice_thickness_mm=resolved_slice_thickness_mm,
        slice_samples=DEFAULT_SLAB_SAMPLES,
        source_oblique_size_mm=resolved_source_oblique_size_mm,
        reference_fov_mm=resolved_reference_fov_mm,
        source_plane="nUS_nB_with_lateral_thickness",
        display_plane="nUS_nB_fan",
        reference_plane="nUS_nB_with_lateral_thickness",
        refine_contact_enabled=bool(refine_contact),
        diagnostic_panel_enabled=overlay_config.diagnostic_panel_enabled,
        diagnostic_panel_layout=panel_layout,
        virtual_ebus_enabled=overlay_config.virtual_ebus_enabled,
        simulated_ebus_enabled=overlay_config.simulated_ebus_enabled,
        contact_world=list(device_pose.contact_refinement.refined_contact_world),
        original_contact_world=list(device_pose.contact_refinement.original_contact_world),
        voxel_refined_contact_world=list(device_pose.contact_refinement.voxel_refined_contact_world),
        refined_contact_world=list(device_pose.contact_refinement.refined_contact_world),
        tip_start_world=list(device_pose.tip_start_world),
        target_world=list(device_pose.target_world),
        nearest_centerline_point=pose.nearest_centerline_point,
        pose_axes={
            "shaft_axis": (None if pose.shaft_axis is None else [float(value) for value in pose.shaft_axis]),
            "depth_axis": (None if pose.depth_axis is None else [float(value) for value in pose.depth_axis]),
            "lateral_axis": (None if pose.lateral_axis is None else [float(value) for value in pose.lateral_axis]),
        },
        device_axes={
            "nB": list(device_pose.shaft_axis_world),
            "nC": list(device_pose.video_axis_world),
            "nUS": list(device_pose.probe_axis_world),
            "wall_normal": list(device_pose.wall_normal_world),
        },
        target_in_default_forward_hemisphere=pose.target_in_default_forward_hemisphere,
        contact_to_airway_distance_mm=float(device_pose.contact_refinement.refined_contact_to_airway_distance_mm),
        original_contact_to_airway_distance_mm=float(device_pose.contact_refinement.original_contact_to_airway_distance_mm),
        voxel_refined_contact_to_airway_distance_mm=float(device_pose.contact_refinement.voxel_refined_contact_to_airway_distance_mm),
        refined_contact_to_airway_distance_mm=float(device_pose.contact_refinement.refined_contact_to_airway_distance_mm),
        centerline_projection_distance_mm=pose.centerline_projection_distance_mm,
        contact_refinement_method=device_pose.contact_refinement.refinement_method,
        pose_comparison={
            "markup_contact_world": list(device_pose.contact_refinement.original_contact_world),
            "voxel_refined_contact_world": list(device_pose.contact_refinement.voxel_refined_contact_world),
            "mesh_refined_contact_world": list(device_pose.contact_refinement.refined_contact_world),
            "voxel_to_mesh_contact_distance_mm": float(device_pose.contact_refinement.voxel_to_mesh_contact_distance_mm),
            "voxel_contact_to_airway_distance_mm": float(device_pose.contact_refinement.voxel_refined_contact_to_airway_distance_mm),
            "mesh_contact_to_airway_distance_mm": float(device_pose.contact_refinement.refined_contact_to_airway_distance_mm),
            "voxel_refinement_method": device_pose.contact_refinement.voxel_refinement_method,
            "mesh_refinement_method": device_pose.contact_refinement.mesh_refinement_method,
            "voxel_wall_normal_world": list(device_pose.voxel_wall_normal_world),
            "mesh_wall_normal_world": list(device_pose.wall_normal_world),
            "voxel_nUS_world": list(device_pose.voxel_probe_axis_world),
            "mesh_nUS_world": list(device_pose.probe_axis_world),
            "nUS_angular_difference_deg": float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(device_pose.voxel_probe_axis_world, device_pose.probe_axis_world)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "final_nB_world": list(device_pose.shaft_axis_world),
            "final_nUS_world": list(device_pose.probe_axis_world),
            "final_nC_world": list(device_pose.video_axis_world),
            "warnings": list(device_pose.contact_refinement.warnings),
        },
        airway_overlay_enabled=(overlay_config.airway_lumen_enabled or overlay_config.airway_wall_enabled),
        airway_lumen_overlay_enabled=overlay_config.airway_lumen_enabled,
        airway_wall_overlay_enabled=overlay_config.airway_wall_enabled,
        target_overlay_enabled=overlay_config.target_enabled,
        contact_overlay_enabled=overlay_config.contact_enabled,
        station_overlay_enabled=overlay_config.station_enabled,
        vessel_overlay_names=list(overlay_config.vessel_names),
        single_vessel_name=overlay_config.single_vessel_name,
        show_legend=overlay_config.show_legend,
        label_overlays=overlay_config.label_overlays,
        show_frustum=overlay_config.show_frustum,
        cutaway_mode=cutaway_display.mode,
        cutaway_side=cutaway_display.side,
        cutaway_side_source=cutaway_display.side_source,
        cutaway_open_side=cutaway_display.open_side,
        cutaway_depth_mm=cutaway_display.depth_mm,
        cutaway_origin_mode=cutaway_display.origin_mode,
        cutaway_origin=[float(value) for value in cutaway_display.origin_world.tolist()],
        cutaway_normal=[float(value) for value in cutaway_display.normal_world.tolist()],
        cutaway_mesh_source=cutaway_display.mesh_source,
        show_full_airway=cutaway_display.show_full_airway,
        overlays_enabled=_overlay_summary(overlay_config),
        warnings=warnings,
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))

    return RenderedPreset(
        image_rgb=output_image_uint8,
        sector_mask=sector_mask,
        metadata=metadata,
    )


def render_all_presets(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
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
) -> BatchRenderIndex:
    context = build_render_context(manifest_path, roll_deg=roll_deg)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
                context=context,
            )
            metadata = rendered.metadata
            entries.append(
                RenderIndexEntry(
                    preset_id=metadata.preset_id,
                    approach=metadata.approach,
                    mode=metadata.mode,
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
