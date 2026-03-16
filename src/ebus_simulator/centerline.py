from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ebus_simulator.io.vtp import load_vtp_polydata


EPSILON = 1e-9


@dataclass(slots=True)
class CenterlinePolyline:
    line_index: int
    point_indices: np.ndarray
    points_lps: np.ndarray
    cumulative_lengths_mm: np.ndarray

    @property
    def total_length_mm(self) -> float:
        return float(self.cumulative_lengths_mm[-1]) if self.cumulative_lengths_mm.size else 0.0

    def point_at_arc_length(self, arc_length_mm: float) -> np.ndarray:
        if self.points_lps.shape[0] == 1:
            return self.points_lps[0].copy()

        clamped = float(np.clip(arc_length_mm, 0.0, self.total_length_mm))
        if clamped <= 0.0:
            return self.points_lps[0].copy()
        if clamped >= self.total_length_mm:
            return self.points_lps[-1].copy()

        stop = int(np.searchsorted(self.cumulative_lengths_mm, clamped, side="right"))
        segment_index = max(0, min(stop - 1, self.points_lps.shape[0] - 2))
        start_length = float(self.cumulative_lengths_mm[segment_index])
        end_length = float(self.cumulative_lengths_mm[segment_index + 1])
        segment_length = end_length - start_length
        if segment_length <= EPSILON:
            return self.points_lps[segment_index].copy()

        t = (clamped - start_length) / segment_length
        start_point = self.points_lps[segment_index]
        end_point = self.points_lps[segment_index + 1]
        return start_point + (end_point - start_point) * t


@dataclass(slots=True)
class CenterlineProjection:
    graph_name: str
    distance_mm: float
    closest_point_lps: np.ndarray
    tangent_lps: np.ndarray | None
    line_index: int
    segment_index: int
    line_arclength_mm: float


class CenterlineGraph:
    def __init__(self, *, name: str, source_path: str, polylines: list[CenterlinePolyline]) -> None:
        self.name = name
        self.source_path = source_path
        self.polylines = polylines
        self.polylines_by_index = {polyline.line_index: polyline for polyline in polylines}

        segment_starts: list[np.ndarray] = []
        segment_ends: list[np.ndarray] = []
        segment_lengths: list[float] = []
        segment_line_indices: list[int] = []
        segment_indices: list[int] = []
        segment_start_arclengths: list[float] = []

        for polyline in polylines:
            if polyline.points_lps.shape[0] < 2:
                continue
            for segment_index, (start_point, end_point) in enumerate(zip(polyline.points_lps[:-1], polyline.points_lps[1:])):
                segment_length = float(np.linalg.norm(end_point - start_point))
                if segment_length <= EPSILON:
                    continue
                segment_starts.append(start_point)
                segment_ends.append(end_point)
                segment_lengths.append(segment_length)
                segment_line_indices.append(polyline.line_index)
                segment_indices.append(segment_index)
                segment_start_arclengths.append(float(polyline.cumulative_lengths_mm[segment_index]))

        self.segment_starts = np.asarray(segment_starts, dtype=np.float64) if segment_starts else np.empty((0, 3), dtype=np.float64)
        self.segment_ends = np.asarray(segment_ends, dtype=np.float64) if segment_ends else np.empty((0, 3), dtype=np.float64)
        self.segment_lengths = np.asarray(segment_lengths, dtype=np.float64) if segment_lengths else np.empty((0,), dtype=np.float64)
        self.segment_line_indices = np.asarray(segment_line_indices, dtype=np.int64) if segment_line_indices else np.empty((0,), dtype=np.int64)
        self.segment_indices = np.asarray(segment_indices, dtype=np.int64) if segment_indices else np.empty((0,), dtype=np.int64)
        self.segment_start_arclengths = (
            np.asarray(segment_start_arclengths, dtype=np.float64) if segment_start_arclengths else np.empty((0,), dtype=np.float64)
        )

    @classmethod
    def from_vtp(cls, path: str, *, name: str) -> "CenterlineGraph":
        polydata = load_vtp_polydata(path)
        polylines: list[CenterlinePolyline] = []
        for line_index, line in enumerate(polydata.lines):
            point_indices = np.asarray(line, dtype=np.int64)
            points = polydata.points_lps[point_indices]
            if points.shape[0] == 0:
                continue
            if points.shape[0] == 1:
                cumulative = np.asarray([0.0], dtype=np.float64)
            else:
                cumulative = np.concatenate(
                    (
                        np.asarray([0.0], dtype=np.float64),
                        np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
                    )
                )
            polylines.append(
                CenterlinePolyline(
                    line_index=line_index,
                    point_indices=point_indices,
                    points_lps=points.astype(np.float64),
                    cumulative_lengths_mm=cumulative.astype(np.float64),
                )
            )
        return cls(name=name, source_path=str(polydata.path), polylines=polylines)

    @property
    def line_count(self) -> int:
        return len(self.polylines)

    @property
    def point_count(self) -> int:
        return int(sum(polyline.points_lps.shape[0] for polyline in self.polylines))

    @property
    def segment_count(self) -> int:
        return int(self.segment_starts.shape[0])

    def estimate_tangent(self, *, line_index: int, line_arclength_mm: float, window_mm: float = 5.0) -> np.ndarray | None:
        polyline = self.polylines_by_index[int(line_index)]
        if polyline.total_length_mm <= EPSILON:
            return None

        start_s = max(0.0, line_arclength_mm - window_mm)
        end_s = min(polyline.total_length_mm, line_arclength_mm + window_mm)

        if end_s - start_s <= EPSILON:
            start_s = max(0.0, line_arclength_mm - (window_mm * 2.0))
            end_s = min(polyline.total_length_mm, line_arclength_mm + (window_mm * 2.0))

        start_point = polyline.point_at_arc_length(start_s)
        end_point = polyline.point_at_arc_length(end_s)
        tangent = end_point - start_point
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > EPSILON:
            return tangent / tangent_norm

        if polyline.points_lps.shape[0] >= 2:
            fallback = polyline.points_lps[-1] - polyline.points_lps[0]
            fallback_norm = np.linalg.norm(fallback)
            if fallback_norm > EPSILON:
                return fallback / fallback_norm
        return None

    def nearest_point(self, point_lps: np.ndarray, *, tangent_window_mm: float = 5.0) -> CenterlineProjection | None:
        if self.segment_starts.size == 0 or self.segment_ends.size == 0:
            return None

        vectors = self.segment_ends - self.segment_starts
        point_vectors = point_lps - self.segment_starts
        lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
        t = np.einsum("ij,ij->i", point_vectors, vectors) / lengths_sq
        t = np.clip(t, 0.0, 1.0)
        closest = self.segment_starts + vectors * t[:, None]
        distances = np.linalg.norm(closest - point_lps, axis=1)
        best = int(np.argmin(distances))

        line_index = int(self.segment_line_indices[best])
        segment_index = int(self.segment_indices[best])
        line_arclength_mm = float(self.segment_start_arclengths[best] + (t[best] * self.segment_lengths[best]))
        tangent = self.estimate_tangent(line_index=line_index, line_arclength_mm=line_arclength_mm, window_mm=tangent_window_mm)
        if tangent is None:
            segment_vector = vectors[best]
            segment_norm = np.linalg.norm(segment_vector)
            tangent = (segment_vector / segment_norm) if segment_norm > EPSILON else None

        return CenterlineProjection(
            graph_name=self.name,
            distance_mm=float(distances[best]),
            closest_point_lps=closest[best].astype(np.float64),
            tangent_lps=(None if tangent is None else tangent.astype(np.float64)),
            line_index=line_index,
            segment_index=segment_index,
            line_arclength_mm=line_arclength_mm,
        )
