from __future__ import annotations

import numpy as np

from ebus_simulator.poses import _choose_fallback_depth_axis, _orthogonality_check, _rotate_around_axis, _status_from_counts


def test_choose_fallback_depth_axis_returns_normalized_perpendicular_axis():
    shaft_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    depth_axis = _choose_fallback_depth_axis(shaft_axis)

    assert np.isclose(np.linalg.norm(depth_axis), 1.0)
    assert np.isclose(np.dot(depth_axis, shaft_axis), 0.0)


def test_rotate_around_axis_rotates_vector_about_shaft():
    rotated = _rotate_around_axis(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        90.0,
    )

    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)


def test_orthogonality_check_reports_drift():
    check = _orthogonality_check(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        np.asarray([0.2, 0.0, 1.0], dtype=np.float64),
    )

    assert check.within_tolerance is False
    assert check.max_abs_dot > 0.0


def test_status_from_counts_prioritizes_errors_then_warnings():
    assert _status_from_counts(1, 0) == "failed"
    assert _status_from_counts(0, 2) == "warning"
    assert _status_from_counts(0, 0) == "passed"
