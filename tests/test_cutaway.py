from pathlib import Path

import numpy as np

from ebus_simulator.cutaway import build_display_cutaway, default_cutaway_side
from ebus_simulator.models import PolyData


def _simple_mesh() -> PolyData:
    return PolyData(
        path=Path("/tmp/simple.vtp"),
        points_lps=np.asarray(
            [
                [0.0, -2.0, -2.0],
                [0.0, 2.0, -2.0],
                [0.0, 2.0, 2.0],
                [0.0, -2.0, 2.0],
            ],
            dtype=np.float64,
        ),
        lines=[],
        point_data={},
        field_data={},
        source_space="LPS",
        polygons=[np.asarray([0, 1, 2], dtype=np.int64), np.asarray([0, 2, 3], dtype=np.int64)],
    )


def test_default_cutaway_side_rules():
    assert default_cutaway_side("4r", None) == "right"
    assert default_cutaway_side("4l", None) == "left"
    assert default_cutaway_side("7", "lms") == "left"
    assert default_cutaway_side("7", "rms") == "right"
    assert default_cutaway_side("2l", None) == "auto"


def test_build_display_cutaway_uses_lps_right_for_right_sided_station():
    cutaway = build_display_cutaway(
        _simple_mesh(),
        mesh_source="smoothed",
        station="4r",
        approach="default",
        mode="lateral",
        requested_side="auto",
        origin_mode="contact",
        depth_mm=0.0,
        show_full_airway=False,
        contact_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        target_world=np.asarray([-10.0, 4.0, 0.0], dtype=np.float64),
        lateral_axis_world=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        probe_axis_world=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        shaft_axis_world=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        station_visibility_points_world=np.asarray([[-12.0, 3.0, 0.0]], dtype=np.float64),
    )

    assert cutaway.side == "right"
    assert cutaway.open_side == "right"
    assert cutaway.normal_world[0] < 0.0
    assert cutaway.mesh_source == "smoothed"


def test_build_display_cutaway_distinguishes_station_7_approaches():
    common_kwargs = dict(
        mesh=_simple_mesh(),
        mesh_source="smoothed",
        station="7",
        mode="lateral",
        requested_side="auto",
        origin_mode="contact",
        depth_mm=0.0,
        show_full_airway=False,
        contact_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        target_world=np.asarray([0.0, 8.0, 0.0], dtype=np.float64),
        lateral_axis_world=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        probe_axis_world=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        shaft_axis_world=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )

    lms = build_display_cutaway(approach="lms", **common_kwargs)
    rms = build_display_cutaway(approach="rms", **common_kwargs)

    assert lms.side == "left"
    assert rms.side == "right"
    assert lms.normal_world[0] > 0.0
    assert rms.normal_world[0] < 0.0
