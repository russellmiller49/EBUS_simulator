from pathlib import Path

import numpy as np
import pytest

from ebus_simulator.poses import generate_pose_report, pose_report_to_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


@pytest.fixture(scope="module")
def pose_report():
    return generate_pose_report(MANIFEST_PATH)


def test_pose_report_covers_all_contact_approaches(pose_report):
    report = pose_report
    assert report.case_id == "3D_slicer_files"
    assert report.preset_count == 15
    assert report.approach_count == 16
    assert all(pose.status != "failed" for pose in report.poses)


def test_pose_axes_are_orthonormal_within_tolerance(pose_report):
    report = pose_report
    for pose in report.poses:
        assert pose.orthogonality is not None
        assert pose.orthogonality.within_tolerance
        assert pose.shaft_axis is not None
        assert pose.depth_axis is not None
        assert pose.lateral_axis is not None


def test_station_7_lms_and_rms_yield_different_poses(pose_report):
    report = pose_report
    station_7_poses = {
        pose.contact_approach: pose
        for pose in report.poses
        if pose.preset_id == "station_7_node_a"
    }
    assert set(station_7_poses) == {"lms", "rms"}
    assert not np.allclose(station_7_poses["lms"].contact_world, station_7_poses["rms"].contact_world)
    assert not np.allclose(station_7_poses["lms"].shaft_axis, station_7_poses["rms"].shaft_axis)


def test_pose_generation_is_deterministic():
    first = pose_report_to_dict(generate_pose_report(MANIFEST_PATH))
    second = pose_report_to_dict(generate_pose_report(MANIFEST_PATH))
    assert first == second
