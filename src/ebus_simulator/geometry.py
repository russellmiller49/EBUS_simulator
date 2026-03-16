from __future__ import annotations

import numpy as np
from scipy import ndimage

from ebus_simulator.models import PolyData, VolumeData


_DISTANCE_MAP_CACHE: dict[tuple[str, tuple[int, int, int], tuple[float, float, float]], np.ndarray] = {}
_SIGNED_DISTANCE_CACHE: dict[tuple[str, tuple[int, int, int], tuple[float, float, float]], np.ndarray] = {}


def point_to_voxel(point_lps: np.ndarray, volume: VolumeData) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point_lps
    ijk = volume.inverse_affine_lps @ homogeneous
    return ijk[:3]


def points_to_voxels(points_lps: np.ndarray, volume: VolumeData) -> np.ndarray:
    points = np.asarray(points_lps, dtype=np.float64)
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    return (homogeneous @ volume.inverse_affine_lps.T)[:, :3]


def point_inside_volume(point_lps: np.ndarray, volume: VolumeData, *, margin_voxels: float = 0.5) -> bool:
    ijk = point_to_voxel(point_lps, volume)
    limits = np.asarray(volume.shape[:3], dtype=np.float64) - 1.0
    return bool(np.all(ijk >= -margin_voxels) and np.all(ijk <= limits + margin_voxels))


def points_inside_volume(points_lps: np.ndarray, volume: VolumeData, *, margin_voxels: float = 0.5) -> np.ndarray:
    ijk = points_to_voxels(points_lps, volume)
    limits = np.asarray(volume.shape[:3], dtype=np.float64) - 1.0
    return np.all((ijk >= -margin_voxels) & (ijk <= (limits + margin_voxels)), axis=1)


def sample_volume_scalar(point_lps: np.ndarray, volume: VolumeData, *, order: int, cval: float) -> float:
    if volume.data is None:
        raise ValueError(f"Volume {volume.path} must be loaded for scalar sampling.")
    voxel = point_to_voxel(point_lps, volume)
    value = ndimage.map_coordinates(
        np.asarray(volume.data, dtype=np.float32),
        [[float(voxel[0])], [float(voxel[1])], [float(voxel[2])]],
        order=order,
        mode="constant",
        cval=cval,
    )
    return float(value[0])


def nearest_index(point_lps: np.ndarray, volume: VolumeData) -> tuple[int, int, int]:
    ijk = point_to_voxel(point_lps, volume)
    rounded = np.rint(ijk).astype(int)
    clipped = np.clip(rounded, [0, 0, 0], np.asarray(volume.shape[:3]) - 1)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def mask_contains_point(point_lps: np.ndarray, mask_volume: VolumeData) -> bool:
    if mask_volume.data is None:
        raise ValueError("Mask data is required for containment checks.")
    index = nearest_index(point_lps, mask_volume)
    return bool(mask_volume.data[index] > 0)


def mask_contains_points(points_lps: np.ndarray, mask_volume: VolumeData) -> np.ndarray:
    if mask_volume.data is None:
        raise ValueError("Mask data is required for containment checks.")
    ijk = np.rint(points_to_voxels(points_lps, mask_volume)).astype(int)
    clipped = np.clip(ijk, [0, 0, 0], np.asarray(mask_volume.shape[:3]) - 1)
    return np.asarray(mask_volume.data[clipped[:, 0], clipped[:, 1], clipped[:, 2]] > 0, dtype=bool)


def _surface_distance_map(mask_key: str, mask: np.ndarray, shape: tuple[int, int, int], voxel_sizes: tuple[float, float, float]) -> np.ndarray:
    cache_key = (mask_key, shape, voxel_sizes)
    cached = _DISTANCE_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    eroded = ndimage.binary_erosion(mask, border_value=0)
    surface = mask ^ eroded
    if not np.any(surface):
        distance_map = ndimage.distance_transform_edt(~mask, sampling=voxel_sizes)
    else:
        distance_map = ndimage.distance_transform_edt(~surface, sampling=voxel_sizes)
    _DISTANCE_MAP_CACHE[cache_key] = distance_map
    return distance_map


def distance_to_mask_surface_mm(point_lps: np.ndarray, mask_volume: VolumeData) -> float:
    if mask_volume.data is None:
        raise ValueError("Mask data is required for distance checks.")
    mask = np.asarray(mask_volume.data > 0, dtype=np.uint8)
    distance_map = _surface_distance_map(
        str(mask_volume.path),
        mask > 0,
        tuple(int(value) for value in mask.shape[:3]),
        tuple(float(value) for value in mask_volume.voxel_sizes_mm[:3]),
    )
    index = nearest_index(point_lps, mask_volume)
    return float(distance_map[index])


def distance_to_mask_surface_points_mm(points_lps: np.ndarray, mask_volume: VolumeData) -> np.ndarray:
    if mask_volume.data is None:
        raise ValueError("Mask data is required for distance checks.")
    mask = np.asarray(mask_volume.data > 0, dtype=np.uint8)
    distance_map = _surface_distance_map(
        str(mask_volume.path),
        mask > 0,
        tuple(int(value) for value in mask.shape[:3]),
        tuple(float(value) for value in mask_volume.voxel_sizes_mm[:3]),
    )
    ijk = np.rint(points_to_voxels(points_lps, mask_volume)).astype(int)
    clipped = np.clip(ijk, [0, 0, 0], np.asarray(mask_volume.shape[:3]) - 1)
    return np.asarray(distance_map[clipped[:, 0], clipped[:, 1], clipped[:, 2]], dtype=np.float64)


def signed_distance_volume(mask_volume: VolumeData) -> np.ndarray:
    if mask_volume.data is None:
        raise ValueError("Mask data is required for signed-distance computation.")
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
    signed_distance = (outside - inside).astype(np.float32)
    _SIGNED_DISTANCE_CACHE[cache_key] = signed_distance
    return signed_distance


def sample_signed_distance_mm(point_lps: np.ndarray, mask_volume: VolumeData) -> float:
    signed_distance = signed_distance_volume(mask_volume)
    proxy = VolumeData(
        path=mask_volume.path,
        kind=mask_volume.kind,
        shape=mask_volume.shape,
        dtype="float32",
        affine_ras=mask_volume.affine_ras,
        affine_lps=mask_volume.affine_lps,
        inverse_affine_lps=mask_volume.inverse_affine_lps,
        voxel_sizes_mm=mask_volume.voxel_sizes_mm,
        axis_codes_ras=mask_volume.axis_codes_ras,
        data=signed_distance,
    )
    return sample_volume_scalar(point_lps, proxy, order=1, cval=float(np.max(signed_distance)))


def build_centerline_segments(polydata: PolyData) -> tuple[np.ndarray, np.ndarray]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for line in polydata.lines:
        for first, second in zip(line[:-1], line[1:]):
            start = polydata.points_lps[int(first)]
            end = polydata.points_lps[int(second)]
            if np.allclose(start, end):
                continue
            starts.append(start)
            ends.append(end)
    if not starts:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)


def project_point_to_segments(point_lps: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> dict[str, object] | None:
    if starts.size == 0 or ends.size == 0:
        return None

    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    valid = lengths_sq > 0
    starts = starts[valid]
    ends = ends[valid]
    vectors = vectors[valid]
    lengths_sq = lengths_sq[valid]
    if starts.size == 0:
        return None

    point_vectors = point_lps - starts
    t = np.einsum("ij,ij->i", point_vectors, vectors) / lengths_sq
    t = np.clip(t, 0.0, 1.0)
    closest = starts + vectors * t[:, None]
    deltas = closest - point_lps
    distances = np.linalg.norm(deltas, axis=1)
    best = int(np.argmin(distances))

    tangent = vectors[best]
    tangent_norm = np.linalg.norm(tangent)
    tangent_unit = (tangent / tangent_norm) if tangent_norm > 0 else None

    return {
        "distance_mm": float(distances[best]),
        "closest_point_lps": closest[best].tolist(),
        "tangent_lps": tangent_unit.tolist() if tangent_unit is not None else None,
        "tangent_defined": tangent_unit is not None,
    }
