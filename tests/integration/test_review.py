from __future__ import annotations

import json
from pathlib import Path

from ebus_simulator.review import review_presets


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_review_presets_generates_physics_aware_bundle(tmp_path):
    output_dir = tmp_path / "review_bundle"
    summary = review_presets(
        MANIFEST_PATH,
        output_dir=output_dir,
        width=64,
        height=64,
        preset_ids=["station_4r_node_b", "station_7_node_a"],
        include_physics_debug_maps=True,
        physics_profile="sparse_support_boost",
        physics_speckle_strength=0.22,
        physics_reverberation_strength=0.28,
        physics_shadow_strength=0.47,
    )

    index_payload = json.loads((output_dir / "review_index.json").read_text())
    index_markdown = (output_dir / "review_index.md").read_text()
    entry_keys = [(entry["preset_id"], entry["approach"]) for entry in summary["entries"]]

    assert summary["review_count"] == 3
    assert summary["include_physics_debug_maps"] is True
    assert (output_dir / "review_index.json").exists()
    assert (output_dir / "review_index.csv").exists()
    assert (output_dir / "review_index.md").exists()
    assert (output_dir / "review_rubric_template.md").exists()
    assert entry_keys == [
        ("station_4r_node_b", "default"),
        ("station_7_node_a", "lms"),
        ("station_7_node_a", "rms"),
    ]
    assert {entry["approach"] for entry in summary["entries"] if entry["preset_id"] == "station_7_node_a"} == {"lms", "rms"}
    assert all(Path(entry["localizer_panel_png"]).exists() for entry in summary["entries"])
    assert all(Path(entry["physics_png"]).exists() for entry in summary["entries"])
    assert all(Path(entry["physics_json"]).exists() for entry in summary["entries"])
    assert all(Path(entry["eval_summary_json"]).exists() for entry in summary["entries"])
    assert all(Path(entry["review_sheet_md"]).exists() for entry in summary["entries"])
    assert all(entry["physics_debug_map_count"] > 0 for entry in summary["entries"])
    assert all(
        json.loads(Path(entry["eval_summary_json"]).read_text())["eval_summary"] == entry["physics_eval_summary"]
        for entry in summary["entries"]
    )
    assert all(
        set(entry["physics_artifact_settings"]) == {"speckle_strength", "reverberation_strength", "shadow_strength"}
        for entry in summary["entries"]
    )
    assert summary["physics_settings"]["profile_name"] == "sparse_support_boost"
    assert summary["physics_settings"]["explicit_overrides"] == {
        "speckle_strength": 0.22,
        "reverberation_strength": 0.28,
        "shadow_strength": 0.47,
    }
    assert all(entry["physics_profile"]["name"] == "sparse_support_boost" for entry in summary["entries"])
    assert all(
        json.loads(Path(entry["eval_summary_json"]).read_text())["profile"]["name"] == "sparse_support_boost"
        for entry in summary["entries"]
    )
    assert summary["thresholds"]["target_contrast_vs_sector_min"] == 0.0
    assert summary["thresholds"]["vessel_contrast_vs_sector_max"] == -0.01
    assert summary["thresholds"]["wall_contrast_vs_sector_min"] == 0.02
    assert all("geometry_flag_reasons" in entry for entry in summary["entries"])
    assert all("consistency_flag_reasons" in entry for entry in summary["entries"])
    assert all("physics_flag_reasons" in entry for entry in summary["entries"])
    assert all("target_contrast_vs_sector" in entry["physics_eval_summary"] for entry in summary["entries"])
    assert all("localizer_consistency_metrics" in entry for entry in summary["entries"])
    assert all("physics_consistency_metrics" in entry for entry in summary["entries"])
    assert all("consistency_bucket" in entry["physics_consistency_metrics"] for entry in summary["entries"])
    assert all("support_logic_active" in entry["physics_consistency_metrics"] for entry in summary["entries"])
    assert any(entry["physics_eval_summary"]["wall"]["pixel_count"] > 0 for entry in summary["entries"])
    assert index_payload["review_count"] == 3
    assert "| station_7_node_a | lms |" in index_markdown
    assert "| station_7_node_a | rms |" in index_markdown
