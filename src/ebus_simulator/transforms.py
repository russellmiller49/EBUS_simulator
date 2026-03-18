from __future__ import annotations

import numpy as np
from scipy import ndimage


OPTIMIZATION_EPSILON = 1e-9
DEFAULT_SLAB_SAMPLES = 5
WINDOW_CENTER_HU = 40.0
WINDOW_WIDTH_HU = 300.0
SOURCE_SECTION_PREPROCESS_BLEND = 0.30


def _points_to_voxel(points_lps: np.ndarray, inverse_affine_lps: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points_lps, np.ones((points_lps.shape[0], 1), dtype=np.float64)), axis=1)
    ijk = homogeneous @ inverse_affine_lps.T
    return ijk[:, :3]


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= OPTIMIZATION_EPSILON:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) - (float(np.dot(vector, axis)) * np.asarray(axis, dtype=np.float64))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = _normalize(a)
    b_unit = _normalize(b)
    if a_unit is None or b_unit is None:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a_unit, b_unit)), -1.0, 1.0))))


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


def _sample_plane(
    context,
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
    del context
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
