from __future__ import annotations

import numpy as np

from ebus_simulator.transforms import _build_sector_grid, _fan_target_row_col, _points_to_voxel, _window_ct


def test_points_to_voxel_respects_inverse_affine():
    points = np.asarray([[10.0, 20.0, 30.0]], dtype=np.float64)
    inverse_affine = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    voxel = _points_to_voxel(points, inverse_affine)

    assert np.allclose(voxel, [[5.0, 10.0, 15.0]])


def test_build_sector_grid_masks_outside_sector_edges():
    depth_grid, lateral_grid, sector_mask, max_lateral_mm = _build_sector_grid(5, 5, 40.0, 60.0)

    assert np.isclose(max_lateral_mm, 40.0 * np.tan(np.deg2rad(30.0)))
    assert sector_mask[0, 2]
    assert not sector_mask[0, 0]
    assert depth_grid.shape == (5, 5)
    assert lateral_grid.shape == (5, 5)


def test_window_ct_applies_depth_attenuation():
    hu = np.asarray([[40.0], [40.0]], dtype=np.float32)
    depths = np.asarray([[0.0], [40.0]], dtype=np.float32)

    windowed = _window_ct(hu, gain=1.0, attenuation=0.5, depths_mm=depths, max_depth_mm=40.0)

    assert np.isclose(windowed[0, 0], 0.5)
    assert windowed[1, 0] < windowed[0, 0]


def test_fan_target_row_col_returns_center_for_straight_ahead_target():
    _, _, _, max_lateral_mm = _build_sector_grid(7, 7, 40.0, 60.0)

    row_col = _fan_target_row_col(
        contact_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        target_world=np.asarray([0.0, 20.0, 0.0], dtype=np.float64),
        probe_axis=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        shaft_axis=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        width=7,
        height=7,
        max_lateral_mm=max_lateral_mm,
    )

    assert row_col == (3, 3)


def test_fan_target_row_col_rejects_targets_behind_probe():
    _, _, _, max_lateral_mm = _build_sector_grid(7, 7, 40.0, 60.0)

    row_col = _fan_target_row_col(
        contact_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        target_world=np.asarray([0.0, -5.0, 0.0], dtype=np.float64),
        probe_axis=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        shaft_axis=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        width=7,
        height=7,
        max_lateral_mm=max_lateral_mm,
    )

    assert row_col is None
