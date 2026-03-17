from ebus_simulator.review import _flag_review_metrics


def test_flag_review_metrics_catches_requested_thresholds():
    flags = _flag_review_metrics(
        {
            "nUS_delta_deg_from_voxel_baseline": 14.5,
            "contact_delta_mm_from_voxel_baseline": 2.1,
            "target_in_sector": False,
            "station_overlap_fraction_in_fan": 0.001,
            "contact_refinement_ambiguity": True,
        }
    )

    assert any("nUS delta" in reason for reason in flags)
    assert any("contact delta" in reason for reason in flags)
    assert any("target not in displayed fan sector" == reason for reason in flags)
    assert any("station overlap" in reason for reason in flags)
    assert any("contact refinement remained ambiguous" == reason for reason in flags)
