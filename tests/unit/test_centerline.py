from __future__ import annotations

import numpy as np

from ebus_simulator.centerline import CenterlineGraph, CenterlinePolyline


def _polyline(points: list[list[float]], *, line_index: int = 0) -> CenterlinePolyline:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.shape[0] <= 1:
        cumulative = np.asarray([0.0], dtype=np.float64)
    else:
        cumulative = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(np.linalg.norm(np.diff(points_array, axis=0), axis=1)),
            )
        )
    return CenterlinePolyline(
        line_index=line_index,
        point_indices=np.arange(points_array.shape[0], dtype=np.int64),
        points_lps=points_array,
        cumulative_lengths_mm=cumulative,
    )


def test_point_at_arc_length_clamps_and_interpolates():
    polyline = _polyline([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]])

    assert np.allclose(polyline.point_at_arc_length(-5.0), [0.0, 0.0, 0.0])
    assert np.allclose(polyline.point_at_arc_length(5.0), [5.0, 0.0, 0.0])
    assert np.allclose(polyline.point_at_arc_length(15.0), [10.0, 5.0, 0.0])
    assert np.allclose(polyline.point_at_arc_length(50.0), [10.0, 10.0, 0.0])


def test_estimate_tangent_handles_short_or_degenerate_windows():
    graph = CenterlineGraph(
        name="main",
        source_path="synthetic.vtp",
        polylines=[_polyline([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])],
    )

    tangent = graph.estimate_tangent(line_index=0, line_arclength_mm=0.0, window_mm=0.0)

    assert tangent is not None
    assert np.allclose(tangent, [1.0, 0.0, 0.0])


def test_nearest_point_returns_closest_segment_and_arclength():
    graph = CenterlineGraph(
        name="main",
        source_path="synthetic.vtp",
        polylines=[_polyline([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]])],
    )

    projection = graph.nearest_point(np.asarray([11.0, 4.0, 0.0], dtype=np.float64))

    assert projection is not None
    assert np.allclose(projection.closest_point_lps, [10.0, 4.0, 0.0])
    assert np.isclose(projection.distance_mm, 1.0)
    assert projection.segment_index == 1
    assert np.isclose(projection.line_arclength_mm, 14.0)
    assert projection.tangent_lps is not None
    assert np.isclose(np.linalg.norm(projection.tangent_lps), 1.0)
    assert projection.tangent_lps[1] > 0.99
    assert projection.tangent_lps[0] > 0.0
