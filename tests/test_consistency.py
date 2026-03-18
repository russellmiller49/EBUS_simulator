from pathlib import Path
import json

from ebus_simulator.consistency import analyze_render_consistency, compare_consistency_summaries


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_analyze_render_consistency_writes_summary_artifacts(tmp_path):
    output_dir = tmp_path / "consistency_bundle"
    summary = analyze_render_consistency(
        MANIFEST_PATH,
        output_dir=output_dir,
        width=48,
        height=48,
        preset_ids=["station_4r_node_b"],
    )

    summary_json = output_dir / "consistency_summary.json"
    summary_csv = output_dir / "consistency_entries.csv"
    summary_md = output_dir / "consistency_summary.md"
    payload = json.loads(summary_json.read_text())

    assert summary["analysis_count"] == 1
    assert summary_json.exists()
    assert summary_csv.exists()
    assert summary_md.exists()
    assert payload["analysis_count"] == 1
    assert payload["entries"][0]["preset_id"] == "station_4r_node_b"
    assert payload["entries"][0]["localizer_consistency_metrics"]["normalization_method"] == "ct_window_interface_blend"
    assert payload["entries"][0]["physics_consistency_metrics"]["normalization_method"].startswith("log_percentile_99.5")
    assert "consistency_bucket" in payload["entries"][0]["physics_consistency_metrics"]
    assert "support_logic_active" in payload["entries"][0]["physics_consistency_metrics"]
    assert "occupancy_gap" in payload["entries"][0]
    assert "representative_cases" in payload


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
