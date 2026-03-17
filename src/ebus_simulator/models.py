from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ManifestPresetOverrides:
    vessel_overlays: list[str] | None = None
    cutaway_side: str | None = None
    roll_offset_deg: float | None = None
    branch_hint: str | None = None
    axis_sign_override: str | None = None
    reference_fov_mm: float | None = None
    notes: str | None = None


@dataclass(slots=True)
class ManifestPreset:
    id: str
    station: str
    node: str
    station_mask: Path
    target: Path
    contacts: dict[str, Path]
    overrides: ManifestPresetOverrides | None = None
    approach_overrides: dict[str, ManifestPresetOverrides] = field(default_factory=dict)


@dataclass(slots=True)
class CaseManifest:
    manifest_path: Path
    case_id: str
    root: Path
    ct_image: Path
    centerline_main: Path
    centerline_network: Path
    primary_markup_curve: Path | None
    secondary_network_curves: list[Path]
    airway_lumen_mask: Path
    airway_solid_mask: Path
    airway_raw_mesh: Path | None
    airway_display_mesh: Path | None
    airway_cutaway_display_mesh: Path | None
    station_masks: dict[str, Path]
    overlay_masks: dict[str, Path]
    presets: list[ManifestPreset]
    qa: dict[str, Any]
    render_defaults: dict[str, Any]
    notes: dict[str, Any]


@dataclass(slots=True)
class MarkupControlPoint:
    id: str
    label: str
    position_lps: np.ndarray
    position_raw: np.ndarray
    position_status: str


@dataclass(slots=True)
class MarkupNode:
    type: str
    coordinate_system: str
    coordinate_units: str | None
    control_points: list[MarkupControlPoint]


@dataclass(slots=True)
class MarkupFile:
    path: Path
    markups: list[MarkupNode]


@dataclass(slots=True)
class VolumeData:
    path: Path
    kind: str
    shape: tuple[int, ...]
    dtype: str
    affine_ras: np.ndarray
    affine_lps: np.ndarray
    inverse_affine_lps: np.ndarray
    voxel_sizes_mm: np.ndarray
    axis_codes_ras: tuple[str, str, str]
    data: np.ndarray | None = None


@dataclass(slots=True)
class PolyData:
    path: Path
    points_lps: np.ndarray
    lines: list[np.ndarray]
    point_data: dict[str, np.ndarray]
    field_data: dict[str, Any]
    source_space: str
    polygons: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    message: str
    preset_id: str | None = None
    approach: str | None = None
    path: str | None = None


@dataclass(slots=True)
class ContactValidation:
    approach: str
    markup_path: str
    point_lps: list[float]
    inside_ct_bounds: bool
    airway_surface_distance_mm: float | None
    raw_mesh_distance_mm: float | None
    centerline_projection_distance_mm: float | None
    tangent_defined: bool
    closest_centerline_point_lps: list[float] | None
    tangent_lps: list[float] | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PresetValidation:
    id: str
    station: str
    node: str
    status: str
    target_markup_path: str
    station_mask_path: str
    target_point_lps: list[float] | None
    target_inside_ct_bounds: bool | None
    target_inside_station_mask: bool | None
    target_station_distance_mm: float | None
    target_to_raw_mesh_signed_distance_mm: float | None
    target_raw_mesh_side: str | None
    target_raw_mesh_side_consistent: bool | None
    contacts: list[ContactValidation]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationReport:
    manifest_path: str
    case_id: str
    dataset_root: str
    internal_world_frame: str
    preset_count: int
    status: str
    ct: dict[str, Any]
    centerlines: dict[str, Any]
    meshes: dict[str, Any]
    issues: list[ValidationIssue]
    presets: list[PresetValidation]


@dataclass(slots=True)
class CenterlineQuery:
    graph_name: str
    distance_mm: float | None
    closest_point_lps: list[float] | None
    tangent_lps: list[float] | None
    tangent_defined: bool
    line_index: int | None
    segment_index: int | None
    line_arclength_mm: float | None


@dataclass(slots=True)
class OrthogonalityCheck:
    shaft_axis_norm: float
    depth_axis_norm: float
    lateral_axis_norm: float
    shaft_depth_dot: float
    shaft_lateral_dot: float
    depth_lateral_dot: float
    max_abs_dot: float
    max_norm_error: float
    within_tolerance: bool


@dataclass(slots=True)
class PresetPose:
    preset_id: str
    station: str
    node: str
    contact_approach: str
    status: str
    contact_markup_path: str
    target_markup_path: str
    contact_world: list[float]
    target_world: list[float]
    contact_to_target_distance_mm: float
    nearest_centerline_point: list[float] | None
    nearest_network_point: list[float] | None
    shaft_axis: list[float] | None
    depth_axis: list[float] | None
    lateral_axis: list[float] | None
    default_depth_axis: list[float] | None
    default_lateral_axis: list[float] | None
    orthogonality: OrthogonalityCheck | None
    target_in_default_forward_hemisphere: bool | None
    target_forward_dot: float | None
    contact_to_airway_distance_mm: float | None
    centerline_projection_distance_mm: float | None
    network_projection_distance_mm: float | None
    centerline_query: CenterlineQuery | None
    network_query: CenterlineQuery | None
    roll_deg: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PoseReport:
    manifest_path: str
    case_id: str
    dataset_root: str
    internal_world_frame: str
    roll_deg: float
    preset_count: int
    approach_count: int
    status: str
    centerlines: dict[str, Any]
    poses: list[PresetPose]
