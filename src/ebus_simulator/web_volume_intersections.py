from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ebus_simulator.centerline import CenterlinePolyline
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.models import VolumeData
from ebus_simulator.web_case_export import ANATOMY_COLORS, DEFAULT_VESSEL_COLOR
from ebus_simulator.web_navigation import (
    WebNavigationPose,
    WebPresetNavigation,
    cephalic_image_axis_lps,
    navigation_pose_from_polyline,
    sector_plane_normal_lps,
)


DEFAULT_DEPTH_SAMPLES = 72
DEFAULT_LATERAL_SAMPLES = 72
DEFAULT_SLAB_HALF_THICKNESS_MM = 2.5
DEFAULT_SLAB_SAMPLES = 5
MIN_HIT_BASE_SAMPLES = 2
EPSILON = 1e-9
PLANE_INTERSECTION_TOLERANCE_MM = 1e-6
MAX_SURFACE_TRIANGLES = 750_000


@dataclass(frozen=True, slots=True)
class SectorSamplingGrid:
    points_lps: np.ndarray
    base_index: np.ndarray
    depth_mm: np.ndarray
    lateral_mm: np.ndarray
    base_sample_count: int
    slab_sample_count: int
    depth_sample_count: int
    lateral_sample_count: int
    slab_half_thickness_mm: float


@dataclass(frozen=True, slots=True)
class WebVolumeMaskSource:
    key: str
    id: str
    kind: str
    label: str
    color: str
    mask_path: Path


@dataclass(frozen=True, slots=True)
class MaskSurfaceMesh:
    path: str
    source: str
    triangle_count: int
    triangles_lps: np.ndarray


@dataclass(frozen=True, slots=True)
class WebVolumeIntersection:
    id: str
    key: str
    kind: str
    label: str
    color: str
    depth_mm: float
    lateral_mm: float
    image_x_mm: float
    normalized_depth: float
    normalized_lateral: float
    visible: bool
    hit_base_sample_count: int
    hit_voxel_sample_count: int
    total_base_sample_count: int
    coverage_fraction: float
    depth_extent_mm: list[float]
    lateral_extent_mm: list[float]
    major_axis_mm: float
    minor_axis_mm: float
    major_axis_vector_mm: list[float]
    aspect_ratio: float
    contours_mm: list[list[list[float]]]
    contour_count: int
    contour_source: str
    surface_source: str
    surface_triangle_count: int
    source: str = "volume_mask"


def _label_from_key(key: str) -> str:
    return key.replace("_", " ").title().replace("Svc", "SVC")


def _station_label(station_key: str) -> str:
    station = station_key.removeprefix("station_").upper()
    return f"Station {station} lymph node"


def _as_positive_int(value: int, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _points_to_voxel(points_lps: np.ndarray, inverse_affine_lps: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points_lps, np.ones((points_lps.shape[0], 1), dtype=np.float64)), axis=1)
    ijk = homogeneous @ inverse_affine_lps.T
    return ijk[:, :3]


def _get_mask_volume(context, mask_path: Path) -> VolumeData:
    key = str(mask_path.resolve())
    cached = context.mask_cache.get(key)
    if cached is not None:
        return cached
    volume = load_nifti(mask_path, kind="mask", load_data=True)
    context.mask_cache[key] = volume
    return volume


def _world_points_from_ijk(points_ijk: np.ndarray, affine_lps: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points_ijk, np.ones((points_ijk.shape[0], 1), dtype=np.float64)), axis=1)
    return (homogeneous @ affine_lps.T)[:, :3]


def _mask_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    nonzero = np.argwhere(mask)
    if nonzero.size == 0:
        return None
    lower = np.maximum(np.min(nonzero, axis=0) - 1, 0)
    upper = np.minimum(np.max(nonzero, axis=0) + 2, np.asarray(mask.shape[:3], dtype=np.int64))
    return lower.astype(np.int64), upper.astype(np.int64)


def _surface_from_marching_cubes(mask: np.ndarray, volume: VolumeData) -> MaskSurfaceMesh | None:
    try:
        from skimage import measure  # type: ignore
    except ImportError:
        return None

    bounds = _mask_bounds(mask)
    if bounds is None:
        return None
    lower, upper = bounds
    cropped = mask[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]].astype(np.float32)
    if not bool(np.any(cropped)):
        return None

    verts_ijk, faces, _, _ = measure.marching_cubes(cropped, level=0.5)
    if faces.size == 0:
        return None
    if int(faces.shape[0]) > MAX_SURFACE_TRIANGLES:
        step = int(math.ceil(int(faces.shape[0]) / MAX_SURFACE_TRIANGLES))
        verts_ijk, faces, _, _ = measure.marching_cubes(cropped, level=0.5, step_size=step)
    verts_lps = _world_points_from_ijk(np.asarray(verts_ijk, dtype=np.float64) + lower[None, :], volume.affine_lps)
    triangles_lps = verts_lps[np.asarray(faces, dtype=np.int64)].astype(np.float32)
    return MaskSurfaceMesh(
        path=str(volume.path),
        source="marching_cubes",
        triangle_count=int(triangles_lps.shape[0]),
        triangles_lps=triangles_lps,
    )


def _shift_mask(mask: np.ndarray, *, axis: int, direction: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    source = [slice(None), slice(None), slice(None)]
    destination = [slice(None), slice(None), slice(None)]
    if direction > 0:
        source[axis] = slice(1, None)
        destination[axis] = slice(0, -1)
    else:
        source[axis] = slice(0, -1)
        destination[axis] = slice(1, None)
    shifted[tuple(destination)] = mask[tuple(source)]
    return shifted


def _face_corners(indices: np.ndarray, *, axis: int, direction: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(indices, dtype=np.float32)
    axis_a, axis_b = [candidate for candidate in range(3) if candidate != axis]
    fixed = 0.5 if direction > 0 else -0.5
    offsets = np.zeros((4, 3), dtype=np.float32)
    offsets[:, axis] = fixed
    offsets[:, axis_a] = [-0.5, 0.5, 0.5, -0.5]
    offsets[:, axis_b] = [-0.5, -0.5, 0.5, 0.5]
    corners = centers[:, None, :] + offsets[None, :, :]
    first = corners[:, [0, 1, 2], :]
    second = corners[:, [0, 2, 3], :]
    return first.reshape((-1, 3)), second.reshape((-1, 3))


def _surface_from_voxel_faces(mask: np.ndarray, volume: VolumeData) -> MaskSurfaceMesh | None:
    triangles_ijk: list[np.ndarray] = []
    for axis in range(3):
        for direction in (-1, 1):
            exposed = mask & ~_shift_mask(mask, axis=axis, direction=direction)
            indices = np.argwhere(exposed)
            if indices.size == 0:
                continue
            first, second = _face_corners(indices, axis=axis, direction=direction)
            triangles_ijk.extend([first, second])

    if not triangles_ijk:
        return None
    stacked_ijk = np.concatenate(triangles_ijk, axis=0)
    if stacked_ijk.shape[0] > MAX_SURFACE_TRIANGLES * 3:
        triangle_count = stacked_ijk.shape[0] // 3
        step = int(math.ceil(triangle_count / MAX_SURFACE_TRIANGLES))
        stacked_ijk = stacked_ijk.reshape((-1, 3, 3))[::step].reshape((-1, 3))
    triangles_lps = _world_points_from_ijk(stacked_ijk.astype(np.float64), volume.affine_lps).reshape((-1, 3, 3)).astype(np.float32)
    return MaskSurfaceMesh(
        path=str(volume.path),
        source="voxel_boundary_faces",
        triangle_count=int(triangles_lps.shape[0]),
        triangles_lps=triangles_lps,
    )


@lru_cache(maxsize=16)
def _cached_surface_mesh(mask_path: str) -> MaskSurfaceMesh | None:
    volume = load_nifti(mask_path, kind="mask", load_data=True)
    if volume.data is None:
        return None
    mask = np.asarray(volume.data) > 0
    if not bool(np.any(mask)):
        return None
    return _surface_from_marching_cubes(mask, volume) or _surface_from_voxel_faces(mask, volume)


def _get_surface_mesh(mask_path: Path) -> MaskSurfaceMesh | None:
    return _cached_surface_mesh(str(mask_path.resolve()))


def build_sector_sampling_grid(
    pose: WebNavigationPose,
    *,
    max_depth_mm: float,
    sector_angle_deg: float,
    depth_samples: int = DEFAULT_DEPTH_SAMPLES,
    lateral_samples: int = DEFAULT_LATERAL_SAMPLES,
    slab_half_thickness_mm: float = DEFAULT_SLAB_HALF_THICKNESS_MM,
    slab_samples: int = DEFAULT_SLAB_SAMPLES,
) -> SectorSamplingGrid:
    resolved_depth_samples = _as_positive_int(depth_samples, default=DEFAULT_DEPTH_SAMPLES, minimum=12, maximum=180)
    resolved_lateral_samples = _as_positive_int(lateral_samples, default=DEFAULT_LATERAL_SAMPLES, minimum=12, maximum=180)
    resolved_slab_samples = _as_positive_int(slab_samples, default=DEFAULT_SLAB_SAMPLES, minimum=1, maximum=11)
    resolved_max_depth = max(float(max_depth_mm), 1.0)
    resolved_slab_half = max(float(slab_half_thickness_mm), 0.0)

    origin = np.asarray(pose.position_lps, dtype=np.float64)
    depth_axis = np.asarray(pose.depth_axis_lps, dtype=np.float64)
    image_axis = cephalic_image_axis_lps(pose)
    plane_normal = sector_plane_normal_lps(pose)
    tan_half_angle = math.tan(math.radians(float(sector_angle_deg) / 2.0))

    first_depth = min(0.5, resolved_max_depth)
    depth_values = np.linspace(first_depth, resolved_max_depth, resolved_depth_samples, dtype=np.float64)
    lateral_fraction = np.linspace(-1.0, 1.0, resolved_lateral_samples, dtype=np.float64)
    lateral_grid = depth_values[:, None] * tan_half_angle * lateral_fraction[None, :]
    depth_grid = np.repeat(depth_values[:, None], resolved_lateral_samples, axis=1)

    base_points = (
        origin[None, None, :]
        + depth_grid[:, :, None] * depth_axis[None, None, :]
        + lateral_grid[:, :, None] * image_axis[None, None, :]
    )
    base_flat = base_points.reshape((-1, 3))
    base_sample_count = int(base_flat.shape[0])

    if resolved_slab_samples <= 1 or resolved_slab_half <= EPSILON:
        slab_offsets = np.asarray([0.0], dtype=np.float64)
    else:
        slab_offsets = np.linspace(-resolved_slab_half, resolved_slab_half, resolved_slab_samples, dtype=np.float64)
    points = base_flat[None, :, :] + slab_offsets[:, None, None] * plane_normal[None, None, :]

    return SectorSamplingGrid(
        points_lps=points.reshape((-1, 3)),
        base_index=np.tile(np.arange(base_sample_count, dtype=np.int64), slab_offsets.shape[0]),
        depth_mm=depth_grid.reshape(-1),
        lateral_mm=lateral_grid.reshape(-1),
        base_sample_count=base_sample_count,
        slab_sample_count=int(slab_offsets.shape[0]),
        depth_sample_count=resolved_depth_samples,
        lateral_sample_count=resolved_lateral_samples,
        slab_half_thickness_mm=float(resolved_slab_half),
    )


def _sample_mask_hits(volume: VolumeData, points_lps: np.ndarray) -> np.ndarray:
    if volume.data is None or points_lps.size == 0:
        return np.zeros(points_lps.shape[0], dtype=bool)

    data = np.asarray(volume.data)
    voxels = _points_to_voxel(points_lps, volume.inverse_affine_lps)
    ijk = np.rint(voxels).astype(np.int64)
    shape = np.asarray(data.shape[:3], dtype=np.int64)
    inside = np.all((ijk >= 0) & (ijk < shape[None, :]), axis=1)
    hits = np.zeros(points_lps.shape[0], dtype=bool)
    if not bool(np.any(inside)):
        return hits

    inside_indices = np.flatnonzero(inside)
    inside_ijk = ijk[inside_indices]
    hits[inside_indices] = data[inside_ijk[:, 0], inside_ijk[:, 1], inside_ijk[:, 2]] > 0
    return hits


_CASE_SEGMENTS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (("left", "top"),),
    2: (("top", "right"),),
    3: (("left", "right"),),
    4: (("right", "bottom"),),
    5: (("left", "bottom"), ("top", "right")),
    6: (("top", "bottom"),),
    7: (("left", "bottom"),),
    8: (("bottom", "left"),),
    9: (("top", "bottom"),),
    10: (("top", "left"), ("right", "bottom")),
    11: (("right", "bottom"),),
    12: (("right", "left"),),
    13: (("top", "right"),),
    14: (("left", "top"),),
}


def _edge_point(edge: str, lateral_grid: np.ndarray, depth_grid: np.ndarray, row: int, column: int) -> tuple[float, float]:
    if edge == "top":
        corners = ((row, column), (row, column + 1))
    elif edge == "right":
        corners = ((row, column + 1), (row + 1, column + 1))
    elif edge == "bottom":
        corners = ((row + 1, column), (row + 1, column + 1))
    elif edge == "left":
        corners = ((row, column), (row + 1, column))
    else:
        raise ValueError(f"Unknown contour edge: {edge}")

    (row_a, column_a), (row_b, column_b) = corners
    return (
        float((lateral_grid[row_a, column_a] + lateral_grid[row_b, column_b]) / 2.0),
        float((depth_grid[row_a, column_a] + depth_grid[row_b, column_b]) / 2.0),
    )


def _segment_key(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0] * 1000.0)), int(round(point[1] * 1000.0)))


def _chain_segments(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    endpoint_map: dict[tuple[int, int], set[int]] = {}
    for index, (start, end) in enumerate(segments):
        endpoint_map.setdefault(_segment_key(start), set()).add(index)
        endpoint_map.setdefault(_segment_key(end), set()).add(index)

    unused = set(range(len(segments)))
    polylines: list[list[tuple[float, float]]] = []

    def pop_connected(endpoint: tuple[float, float]) -> tuple[int, tuple[float, float]] | None:
        candidates = endpoint_map.get(_segment_key(endpoint), set()) & unused
        if not candidates:
            return None
        segment_index = min(candidates)
        unused.remove(segment_index)
        start, end = segments[segment_index]
        return segment_index, (end if _segment_key(start) == _segment_key(endpoint) else start)

    while unused:
        segment_index = min(unused)
        unused.remove(segment_index)
        start, end = segments[segment_index]
        polyline = [start, end]

        while True:
            next_item = pop_connected(polyline[-1])
            if next_item is None:
                break
            _, next_point = next_item
            polyline.append(next_point)
            if _segment_key(polyline[-1]) == _segment_key(polyline[0]):
                break

        while _segment_key(polyline[-1]) != _segment_key(polyline[0]):
            next_item = pop_connected(polyline[0])
            if next_item is None:
                break
            _, next_point = next_item
            polyline.insert(0, next_point)

        polylines.append(polyline)

    return polylines


def _smooth_polyline(points: list[tuple[float, float]], *, iterations: int = 2) -> list[tuple[float, float]]:
    if len(points) < 4:
        return points

    closed = _segment_key(points[0]) == _segment_key(points[-1])
    working = points[:-1] if closed else points[:]
    for _ in range(max(0, int(iterations))):
        if len(working) < 3:
            break
        smoothed: list[tuple[float, float]] = []
        pair_count = len(working) if closed else len(working) - 1
        if not closed:
            smoothed.append(working[0])
        for index in range(pair_count):
            start = working[index]
            end = working[(index + 1) % len(working)]
            q = (0.75 * start[0] + 0.25 * end[0], 0.75 * start[1] + 0.25 * end[1])
            r = (0.25 * start[0] + 0.75 * end[0], 0.25 * start[1] + 0.75 * end[1])
            smoothed.extend([q, r])
        if not closed:
            smoothed.append(working[-1])
        working = smoothed
    if closed:
        working.append(working[0])
    return working


def _contour_score(polyline: list[tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return 0.0
    laterals = [point[0] for point in polyline]
    depths = [point[1] for point in polyline]
    area_box = (max(laterals) - min(laterals)) * (max(depths) - min(depths))
    length = sum(math.dist(polyline[index], polyline[index + 1]) for index in range(len(polyline) - 1))
    return float(area_box + length)


def _contours_from_hit_base_mask(hit_base_mask: np.ndarray, grid: SectorSamplingGrid) -> list[list[list[float]]]:
    if int(np.count_nonzero(hit_base_mask)) < 4:
        return []

    mask = np.asarray(hit_base_mask, dtype=bool).reshape((grid.depth_sample_count, grid.lateral_sample_count))
    depth_grid = grid.depth_mm.reshape((grid.depth_sample_count, grid.lateral_sample_count))
    lateral_grid = grid.lateral_mm.reshape((grid.depth_sample_count, grid.lateral_sample_count))

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in range(mask.shape[0] - 1):
        for column in range(mask.shape[1] - 1):
            case_index = (
                (1 if mask[row, column] else 0)
                | (2 if mask[row, column + 1] else 0)
                | (4 if mask[row + 1, column + 1] else 0)
                | (8 if mask[row + 1, column] else 0)
            )
            for start_edge, end_edge in _CASE_SEGMENTS.get(case_index, ()):
                start = _edge_point(start_edge, lateral_grid, depth_grid, row, column)
                end = _edge_point(end_edge, lateral_grid, depth_grid, row, column)
                if _segment_key(start) != _segment_key(end):
                    segments.append((start, end))

    polylines = _chain_segments(segments)
    filtered = [
        _smooth_polyline(polyline)
        for polyline in polylines
        if len(polyline) >= 4 and _contour_score(polyline) >= 2.0
    ]
    filtered.sort(key=_contour_score, reverse=True)
    return [
        [[float(lateral), float(depth)] for lateral, depth in polyline]
        for polyline in filtered[:4]
    ]


def _unique_points(points: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - existing)) <= 1e-5 for existing in unique):
            unique.append(point)
    return unique


def _farthest_pair(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    best_pair = (points[0], points[1])
    best_distance = -1.0
    for start_index, start in enumerate(points):
        for end in points[start_index + 1 :]:
            distance = float(np.linalg.norm(start - end))
            if distance > best_distance:
                best_distance = distance
                best_pair = (start, end)
    return best_pair


def _triangle_plane_segment(
    triangle: np.ndarray,
    distances: np.ndarray,
    *,
    tolerance_mm: float = PLANE_INTERSECTION_TOLERANCE_MM,
) -> tuple[np.ndarray, np.ndarray] | None:
    if bool(np.all(np.abs(distances) <= tolerance_mm)):
        return None

    points: list[np.ndarray] = []
    for edge_start, edge_end in ((0, 1), (1, 2), (2, 0)):
        d0 = float(distances[edge_start])
        d1 = float(distances[edge_end])
        p0 = triangle[edge_start]
        p1 = triangle[edge_end]
        if abs(d0) <= tolerance_mm:
            points.append(p0)
        if d0 * d1 < 0.0:
            fraction = d0 / (d0 - d1)
            points.append(p0 + fraction * (p1 - p0))
        if abs(d1) <= tolerance_mm:
            points.append(p1)

    pair = _farthest_pair(_unique_points(points))
    if pair is None:
        return None
    if float(np.linalg.norm(pair[0] - pair[1])) <= tolerance_mm:
        return None
    return pair


def _surface_contours_from_mesh(
    surface_mesh: MaskSurfaceMesh,
    pose: WebNavigationPose,
    *,
    max_depth_mm: float,
    sector_angle_deg: float,
) -> list[list[list[float]]]:
    triangles = np.asarray(surface_mesh.triangles_lps, dtype=np.float64)
    if triangles.size == 0:
        return []

    origin = np.asarray(pose.position_lps, dtype=np.float64)
    depth_axis = np.asarray(pose.depth_axis_lps, dtype=np.float64)
    image_axis = cephalic_image_axis_lps(pose)
    plane_normal = sector_plane_normal_lps(pose)
    distances = np.tensordot(triangles - origin[None, None, :], plane_normal, axes=([2], [0]))
    candidate_mask = (np.min(distances, axis=1) <= PLANE_INTERSECTION_TOLERANCE_MM) & (
        np.max(distances, axis=1) >= -PLANE_INTERSECTION_TOLERANCE_MM
    )
    candidate_mask &= np.max(np.abs(distances), axis=1) > PLANE_INTERSECTION_TOLERANCE_MM
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return []

    tan_half_angle = math.tan(math.radians(float(sector_angle_deg) / 2.0))
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index in candidate_indices:
        segment = _triangle_plane_segment(triangles[index], distances[index])
        if segment is None:
            continue
        points_2d: list[tuple[float, float]] = []
        for point_lps in segment:
            offset = point_lps - origin
            lateral_mm = float(np.dot(offset, image_axis))
            depth_mm = float(np.dot(offset, depth_axis))
            points_2d.append((lateral_mm, depth_mm))

        midpoint_lateral = (points_2d[0][0] + points_2d[1][0]) / 2.0
        midpoint_depth = (points_2d[0][1] + points_2d[1][1]) / 2.0
        half_width = max(0.0, midpoint_depth) * tan_half_angle
        margin_mm = 3.0
        if not (-margin_mm <= midpoint_depth <= max_depth_mm + margin_mm):
            continue
        if abs(midpoint_lateral) > half_width + margin_mm:
            continue
        segments.append((points_2d[0], points_2d[1]))

    polylines = _chain_segments(segments)
    filtered = [
        _smooth_polyline(polyline)
        for polyline in polylines
        if len(polyline) >= 4 and _contour_score(polyline) >= 2.0
    ]
    filtered.sort(key=_contour_score, reverse=True)
    return [
        [[float(lateral), float(depth)] for lateral, depth in polyline]
        for polyline in filtered[:4]
    ]


def _intersection_from_hits(
    source: WebVolumeMaskSource,
    *,
    hits: np.ndarray,
    grid: SectorSamplingGrid,
    pose: WebNavigationPose,
    max_depth_mm: float,
    sector_angle_deg: float,
    surface_mesh: MaskSurfaceMesh | None,
    min_hit_base_samples: int,
) -> WebVolumeIntersection | None:
    if not bool(np.any(hits)):
        return None

    hit_base_counts = np.bincount(grid.base_index[hits], minlength=grid.base_sample_count)
    hit_base_mask = hit_base_counts > 0
    hit_base_count = int(np.count_nonzero(hit_base_mask))
    if hit_base_count < int(min_hit_base_samples):
        return None

    surface_contours = (
        []
        if surface_mesh is None
        else _surface_contours_from_mesh(
            surface_mesh,
            pose,
            max_depth_mm=max_depth_mm,
            sector_angle_deg=sector_angle_deg,
        )
    )
    fallback_contours = _contours_from_hit_base_mask(hit_base_mask, grid)
    contours = surface_contours or fallback_contours
    contour_source = "surface_triangle_plane_intersection" if surface_contours else "marching_squares_fan_slice"
    depths = grid.depth_mm[hit_base_mask]
    laterals = grid.lateral_mm[hit_base_mask]
    weights = hit_base_counts[hit_base_mask].astype(np.float64)
    depth_mm = float(np.average(depths, weights=weights))
    lateral_mm = float(np.average(laterals, weights=weights))
    normalized_lateral_denominator = max(float(max_depth_mm), 1.0)

    coords = np.column_stack([laterals, depths])
    centered = coords - np.asarray([lateral_mm, depth_mm], dtype=np.float64)
    if coords.shape[0] >= 3 and float(np.sum(weights)) > EPSILON:
        covariance = (centered * weights[:, None]).T @ centered / float(np.sum(weights))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        major_vector = eigenvectors[:, int(order[0])]
        minor_vector = eigenvectors[:, int(order[1])]
    else:
        lateral_span = float(np.max(laterals) - np.min(laterals)) if laterals.size else 0.0
        depth_span = float(np.max(depths) - np.min(depths)) if depths.size else 0.0
        major_vector = np.asarray([1.0, 0.0], dtype=np.float64) if lateral_span >= depth_span else np.asarray([0.0, 1.0], dtype=np.float64)
        minor_vector = np.asarray([-major_vector[1], major_vector[0]], dtype=np.float64)

    projected_major = centered @ major_vector
    projected_minor = centered @ minor_vector
    major_axis_mm = float(np.max(projected_major) - np.min(projected_major)) if projected_major.size else 0.0
    minor_axis_mm = float(np.max(projected_minor) - np.min(projected_minor)) if projected_minor.size else 0.0
    lateral_span = float(np.max(laterals) - np.min(laterals)) if laterals.size else 0.0
    depth_span = float(np.max(depths) - np.min(depths)) if depths.size else 0.0
    major_axis_mm = max(major_axis_mm, max(lateral_span, depth_span), 1.0)
    minor_axis_mm = max(minor_axis_mm, 1.0)
    if minor_axis_mm > major_axis_mm:
        major_axis_mm, minor_axis_mm = minor_axis_mm, major_axis_mm
        major_vector = minor_vector
    aspect_ratio = float(major_axis_mm / max(minor_axis_mm, 0.5))

    return WebVolumeIntersection(
        id=source.id,
        key=source.key,
        kind=source.kind,
        label=source.label,
        color=source.color,
        depth_mm=depth_mm,
        lateral_mm=lateral_mm,
        image_x_mm=lateral_mm,
        normalized_depth=float(depth_mm / max(float(max_depth_mm), 1.0)),
        normalized_lateral=float(lateral_mm / normalized_lateral_denominator),
        visible=True,
        hit_base_sample_count=hit_base_count,
        hit_voxel_sample_count=int(np.count_nonzero(hits)),
        total_base_sample_count=int(grid.base_sample_count),
        coverage_fraction=float(hit_base_count / max(grid.base_sample_count, 1)),
        depth_extent_mm=[float(np.min(depths)), float(np.max(depths))],
        lateral_extent_mm=[float(np.min(laterals)), float(np.max(laterals))],
        major_axis_mm=major_axis_mm,
        minor_axis_mm=minor_axis_mm,
        major_axis_vector_mm=[float(major_vector[0]), float(major_vector[1])],
        aspect_ratio=aspect_ratio,
        contours_mm=contours,
        contour_count=len(contours),
        contour_source=contour_source,
        surface_source="" if surface_mesh is None else surface_mesh.source,
        surface_triangle_count=0 if surface_mesh is None else int(surface_mesh.triangle_count),
    )


def volume_mask_sources(
    context,
    *,
    station_keys: Iterable[str] | None,
    vessel_keys: Iterable[str] | None,
    color_map: Mapping[str, str] | None = None,
) -> list[WebVolumeMaskSource]:
    colors = dict(color_map or {})
    sources: list[WebVolumeMaskSource] = []

    resolved_station_keys = sorted(set(context.manifest.station_masks.keys() if station_keys is None else station_keys))
    for station_key in resolved_station_keys:
        mask_path = context.manifest.station_masks.get(station_key)
        if mask_path is None:
            continue
        sources.append(
            WebVolumeMaskSource(
                key=station_key,
                id=station_key,
                kind="node",
                label=_station_label(station_key),
                color=str(colors.get("lymph_node") or colors.get("station") or ANATOMY_COLORS["lymph_node"]),
                mask_path=mask_path,
            )
        )

    resolved_vessel_keys = sorted(set(context.manifest.overlay_masks.keys() if vessel_keys is None else vessel_keys))
    for vessel_key in resolved_vessel_keys:
        mask_path = context.manifest.overlay_masks.get(vessel_key)
        if mask_path is None:
            continue
        sources.append(
            WebVolumeMaskSource(
                key=vessel_key,
                id=vessel_key,
                kind="vessel",
                label=_label_from_key(vessel_key),
                color=str(colors.get(vessel_key) or ANATOMY_COLORS.get(vessel_key) or DEFAULT_VESSEL_COLOR),
                mask_path=mask_path,
            )
        )

    return sources


def compute_volume_intersections(
    context,
    *,
    pose: WebNavigationPose,
    sources: Iterable[WebVolumeMaskSource],
    max_depth_mm: float,
    sector_angle_deg: float,
    depth_samples: int = DEFAULT_DEPTH_SAMPLES,
    lateral_samples: int = DEFAULT_LATERAL_SAMPLES,
    slab_half_thickness_mm: float = DEFAULT_SLAB_HALF_THICKNESS_MM,
    slab_samples: int = DEFAULT_SLAB_SAMPLES,
    min_hit_base_samples: int = MIN_HIT_BASE_SAMPLES,
) -> tuple[list[WebVolumeIntersection], SectorSamplingGrid]:
    grid = build_sector_sampling_grid(
        pose,
        max_depth_mm=max_depth_mm,
        sector_angle_deg=sector_angle_deg,
        depth_samples=depth_samples,
        lateral_samples=lateral_samples,
        slab_half_thickness_mm=slab_half_thickness_mm,
        slab_samples=slab_samples,
    )

    intersections: list[WebVolumeIntersection] = []
    for source in sources:
        volume = _get_mask_volume(context, source.mask_path)
        hits = _sample_mask_hits(volume, grid.points_lps)
        surface_mesh = _get_surface_mesh(source.mask_path) if bool(np.any(hits)) else None
        intersection = _intersection_from_hits(
            source,
            hits=hits,
            grid=grid,
            pose=pose,
            max_depth_mm=max_depth_mm,
            sector_angle_deg=sector_angle_deg,
            surface_mesh=surface_mesh,
            min_hit_base_samples=min_hit_base_samples,
        )
        if intersection is not None:
            intersections.append(intersection)

    priority = {"node": 0, "vessel": 1}
    intersections.sort(key=lambda item: (priority.get(item.kind, 99), item.depth_mm, item.label))
    return intersections, grid


def build_volume_sector_response(
    context,
    *,
    centerlines_by_index: Mapping[int, CenterlinePolyline],
    preset: WebPresetNavigation | None,
    line_index: int,
    centerline_s_mm: float,
    roll_deg: float,
    max_depth_mm: float,
    sector_angle_deg: float,
    station_keys: Iterable[str] | None = None,
    vessel_keys: Iterable[str] | None = None,
    color_map: Mapping[str, str] | None = None,
    depth_samples: int = DEFAULT_DEPTH_SAMPLES,
    lateral_samples: int = DEFAULT_LATERAL_SAMPLES,
    slab_half_thickness_mm: float = DEFAULT_SLAB_HALF_THICKNESS_MM,
    slab_samples: int = DEFAULT_SLAB_SAMPLES,
) -> dict[str, object]:
    if line_index not in centerlines_by_index:
        raise ValueError(f"Centerline line_index {line_index} is not available.")

    target_lps = None if preset is None else np.asarray(preset.target_lps, dtype=np.float64)
    pose = navigation_pose_from_polyline(
        centerlines_by_index[line_index],
        centerline_s_mm=centerline_s_mm,
        roll_deg=roll_deg,
        target_lps=target_lps,
    )
    sources = volume_mask_sources(
        context,
        station_keys=station_keys,
        vessel_keys=vessel_keys,
        color_map=color_map,
    )
    intersections, grid = compute_volume_intersections(
        context,
        pose=pose,
        sources=sources,
        max_depth_mm=max_depth_mm,
        sector_angle_deg=sector_angle_deg,
        depth_samples=depth_samples,
        lateral_samples=lateral_samples,
        slab_half_thickness_mm=slab_half_thickness_mm,
        slab_samples=slab_samples,
    )

    return {
        "source": "volume_masks",
        "pose": {
            "line_index": int(pose.line_index),
            "centerline_s_mm": float(pose.centerline_s_mm),
            "roll_deg": float(pose.roll_deg),
            "cephalic_image_axis_lps": [float(value) for value in cephalic_image_axis_lps(pose).tolist()],
        },
        "sector": {
            "max_depth_mm": float(max_depth_mm),
            "sector_angle_deg": float(sector_angle_deg),
            "slab_half_thickness_mm": float(grid.slab_half_thickness_mm),
            "depth_samples": int(grid.depth_sample_count),
            "lateral_samples": int(grid.lateral_sample_count),
            "slab_samples": int(grid.slab_sample_count),
            "labels": [asdict(intersection) for intersection in intersections],
        },
    }
