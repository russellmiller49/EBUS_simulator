from __future__ import annotations

from ebus_simulator.consistency import compare_consistency_summaries


def test_compare_consistency_summaries_tracks_sparse_support_improvements():
    before_summary = {
        "case_id": "demo_case",
        "entries": [
            {
                "preset_id": "station_7_node_a",
                "approach": "lms",
                "occupancy_gap": 0.95,
                "physics_consistency_metrics": {
                    "consistency_bucket": "sparse_empty_dominant",
                    "support_logic_active": False,
                    "empty_sector_fraction": 0.97,
                    "non_background_occupancy_fraction": 0.03,
                    "target_region_contrast_vs_sector": -0.01,
                },
            }
        ],
    }
    after_summary = {
        "case_id": "demo_case",
        "entries": [
            {
                "preset_id": "station_7_node_a",
                "approach": "lms",
                "occupancy_gap": 0.88,
                "physics_consistency_metrics": {
                    "consistency_bucket": "sparse_empty_dominant",
                    "support_logic_active": True,
                    "empty_sector_fraction": 0.93,
                    "non_background_occupancy_fraction": 0.07,
                    "target_region_contrast_vs_sector": 0.02,
                },
            }
        ],
    }

    comparison = compare_consistency_summaries(before_summary, after_summary)

    assert comparison["matched_entry_count"] == 1
    assert comparison["improved_empty_sector_count"] == 1
    assert comparison["improved_non_background_occupancy_count"] == 1
    assert comparison["improved_target_contrast_count"] == 1
    assert comparison["improved_occupancy_gap_count"] == 1
