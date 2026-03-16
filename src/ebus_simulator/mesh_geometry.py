from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ebus_simulator.models import PolyData


EPSILON = 1e-9
DEFAULT_VERTEX_K = 48
DEFAULT_TRIANGLE_K = 24


@dataclass(slots=True)
class MeshQueryResult:
    closest_point_lps: np.ndarray
    distance_mm: float
    triangle_index: int
    barycentric_weights: np.ndarray
    face_normal_lps: np.ndarray | None
    point_normal_lps: np.ndarray | None
    candidate_triangle_count: int


@dataclass(slots=True)
class MeshSurface:
    path: Path
    points_lps: np.ndarray
    triangle_point_indices: np.ndarray
    triangles_lps: np.ndarray
    triangle_centroids_lps: np.ndarray
    triangle_normals_lps: np.ndarray
    vertex_normals_lps: np.ndarray
    vertex_tree: cKDTree
    triangle_centroid_tree: cKDTree
    vertex_to_triangles: tuple[np.ndarray, ...]

    @property
    def point_count(self) -> int:
        return int(self.points_lps.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangle_point_indices.shape[0])

    def nearest_point(
        self,
        point_lps: np.ndarray,
        *,
        vertex_k: int = DEFAULT_VERTEX_K,
        triangle_k: int = DEFAULT_TRIANGLE_K,
    ) -> MeshQueryResult:
        if self.triangle_count == 0:
            raise ValueError(f"Mesh surface {self.path} does not contain any triangles.")

        point = np.asarray(point_lps, dtype=np.float64)
        nearest_vertex_distance, nearest_vertex_indices = self.vertex_tree.query(point, k=min(vertex_k, self.point_count))
        nearest_triangle_distance, nearest_triangle_indices = self.triangle_centroid_tree.query(point, k=min(triangle_k, self.triangle_count))

        candidate_indices = self._collect_candidate_triangles(nearest_vertex_indices, nearest_triangle_indices)
        result = self._evaluate_candidates(point, candidate_indices)

        nearest_vertex_distance = float(np.min(np.atleast_1d(nearest_vertex_distance)))
        if result.distance_mm <= (nearest_vertex_distance + 0.5):
            return result

        expanded_vertex_distance, expanded_vertex_indices = self.vertex_tree.query(point, k=min(self.point_count, max(vertex_k * 2, 96)))
        expanded_triangle_distance, expanded_triangle_indices = self.triangle_centroid_tree.query(
            point,
            k=min(self.triangle_count, max(triangle_k * 2, 48)),
        )
        expanded_candidates = self._collect_candidate_triangles(expanded_vertex_indices, expanded_triangle_indices)
        expanded_result = self._evaluate_candidates(point, expanded_candidates)
        if expanded_result.distance_mm <= (float(np.min(np.atleast_1d(expanded_vertex_distance))) + 0.5):
            return expanded_result

        return self._evaluate_candidates(point, np.arange(self.triangle_count, dtype=np.int64))

    def _collect_candidate_triangles(self, vertex_indices: np.ndarray, triangle_indices: np.ndarray) -> np.ndarray:
        candidates: set[int] = set()
        for vertex_index in np.atleast_1d(vertex_indices).tolist():
            for triangle_index in self.vertex_to_triangles[int(vertex_index)].tolist():
                candidates.add(int(triangle_index))
        for triangle_index in np.atleast_1d(triangle_indices).tolist():
            candidates.add(int(triangle_index))
        if not candidates:
            return np.arange(self.triangle_count, dtype=np.int64)
        return np.asarray(sorted(candidates), dtype=np.int64)

    def _evaluate_candidates(self, point_lps: np.ndarray, candidate_indices: np.ndarray) -> MeshQueryResult:
        best_distance = float("inf")
        best_triangle_index = -1
        best_point = None
        best_barycentric = None

        for triangle_index in np.asarray(candidate_indices, dtype=np.int64).tolist():
            triangle = self.triangles_lps[int(triangle_index)]
            closest_point, barycentric = _closest_point_on_triangle(point_lps, triangle)
            distance = float(np.linalg.norm(closest_point - point_lps))
            if distance < best_distance:
                best_distance = distance
                best_triangle_index = int(triangle_index)
                best_point = closest_point
                best_barycentric = barycentric

        if best_point is None or best_barycentric is None or best_triangle_index < 0:
            raise ValueError(f"Failed to query mesh surface {self.path}.")

        triangle_indices = self.triangle_point_indices[best_triangle_index]
        vertex_normals = self.vertex_normals_lps[triangle_indices]
        interpolated = np.sum(vertex_normals * best_barycentric[:, None], axis=0)
        point_normal = _normalize(interpolated)
        face_normal = self.triangle_normals_lps[best_triangle_index]
        if point_normal is None:
            point_normal = face_normal if np.linalg.norm(face_normal) > EPSILON else None

        return MeshQueryResult(
            closest_point_lps=np.asarray(best_point, dtype=np.float64),
            distance_mm=float(best_distance),
            triangle_index=int(best_triangle_index),
            barycentric_weights=np.asarray(best_barycentric, dtype=np.float64),
            face_normal_lps=(None if np.linalg.norm(face_normal) <= EPSILON else np.asarray(face_normal, dtype=np.float64)),
            point_normal_lps=(None if point_normal is None else np.asarray(point_normal, dtype=np.float64)),
            candidate_triangle_count=int(np.asarray(candidate_indices).size),
        )


_MESH_SURFACE_CACHE: dict[str, MeshSurface] = {}


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _triangulate_polygons(mesh: PolyData) -> np.ndarray:
    triangles: list[list[int]] = []
    for polygon in mesh.polygons:
        point_indices = np.asarray(polygon, dtype=np.int64)
        if point_indices.size < 3:
            continue
        anchor = int(point_indices[0])
        for index in range(1, point_indices.size - 1):
            triangles.append([anchor, int(point_indices[index]), int(point_indices[index + 1])])
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _build_vertex_normals(points_lps: np.ndarray, triangle_point_indices: np.ndarray, triangle_normals_lps: np.ndarray) -> np.ndarray:
    vertex_normals = np.zeros_like(points_lps, dtype=np.float64)
    for triangle_index, point_indices in enumerate(triangle_point_indices):
        normal = triangle_normals_lps[triangle_index]
        if np.linalg.norm(normal) <= EPSILON:
            continue
        for point_index in point_indices.tolist():
            vertex_normals[int(point_index)] += normal
    normalized = np.zeros_like(vertex_normals, dtype=np.float64)
    for point_index in range(vertex_normals.shape[0]):
        normal = _normalize(vertex_normals[point_index])
        if normal is not None:
            normalized[point_index] = normal
    return normalized


def _build_vertex_to_triangles(point_count: int, triangle_point_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    mapping: list[list[int]] = [[] for _ in range(point_count)]
    for triangle_index, point_indices in enumerate(triangle_point_indices):
        for point_index in point_indices.tolist():
            mapping[int(point_index)].append(int(triangle_index))
    return tuple(np.asarray(indices, dtype=np.int64) for indices in mapping)


def _closest_point_on_triangle(point_lps: np.ndarray, triangle_lps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = triangle_lps[0]
    b = triangle_lps[1]
    c = triangle_lps[2]

    ab = b - a
    ac = c - a
    ap = point_lps - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    bp = point_lps - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    vc = (d1 * d4) - (d3 * d2)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, EPSILON)
        return a + (v * ab), np.asarray([1.0 - v, v, 0.0], dtype=np.float64)

    cp = point_lps - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    vb = (d5 * d2) - (d1 * d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, EPSILON)
        return a + (w * ac), np.asarray([1.0 - w, 0.0, w], dtype=np.float64)

    va = (d3 * d6) - (d5 * d4)
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), EPSILON)
        return b + (w * (c - b)), np.asarray([0.0, 1.0 - w, w], dtype=np.float64)

    denom = max(va + vb + vc, EPSILON)
    v = vb / denom
    w = vc / denom
    return a + (ab * v) + (ac * w), np.asarray([1.0 - v - w, v, w], dtype=np.float64)


def get_mesh_surface(mesh: PolyData) -> MeshSurface:
    cache_key = str(mesh.path.resolve())
    cached = _MESH_SURFACE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    triangle_point_indices = _triangulate_polygons(mesh)
    triangles_lps = mesh.points_lps[triangle_point_indices] if triangle_point_indices.size else np.empty((0, 3, 3), dtype=np.float64)
    triangle_normals = np.cross(
        triangles_lps[:, 1] - triangles_lps[:, 0],
        triangles_lps[:, 2] - triangles_lps[:, 0],
    ) if triangles_lps.size else np.empty((0, 3), dtype=np.float64)
    for triangle_index in range(triangle_normals.shape[0]):
        normalized = _normalize(triangle_normals[triangle_index])
        if normalized is None:
            triangle_normals[triangle_index] = 0.0
        else:
            triangle_normals[triangle_index] = normalized

    surface = MeshSurface(
        path=mesh.path,
        points_lps=np.asarray(mesh.points_lps, dtype=np.float64),
        triangle_point_indices=triangle_point_indices,
        triangles_lps=np.asarray(triangles_lps, dtype=np.float64),
        triangle_centroids_lps=(triangles_lps.mean(axis=1) if triangles_lps.size else np.empty((0, 3), dtype=np.float64)),
        triangle_normals_lps=np.asarray(triangle_normals, dtype=np.float64),
        vertex_normals_lps=_build_vertex_normals(np.asarray(mesh.points_lps, dtype=np.float64), triangle_point_indices, triangle_normals),
        vertex_tree=cKDTree(np.asarray(mesh.points_lps, dtype=np.float64)),
        triangle_centroid_tree=cKDTree(triangles_lps.mean(axis=1) if triangles_lps.size else np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)),
        vertex_to_triangles=_build_vertex_to_triangles(int(mesh.points_lps.shape[0]), triangle_point_indices),
    )
    _MESH_SURFACE_CACHE[cache_key] = surface
    return surface
