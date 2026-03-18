from __future__ import annotations

import json
from pathlib import Path

from ebus_simulator.consistency import analyze_render_consistency


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_analyze_render_consistency_writes_summary_artifacts(tmp_path):
    output_dir = tmp_path / "consistency_bundle"
    summary = analyze_render_consistency(
        MANIFEST_PATH,
        output_dir=output_dir,
        width=48,
        height=48,
        preset_ids=["station_4r_node_b"],
        physics_profile="sparse_support_boost",
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
    assert payload["entries"][0]["physics_profile"]["name"] == "sparse_support_boost"
    assert "occupancy_gap" in payload["entries"][0]
    assert "representative_cases" in payload
    assert payload["physics_settings"]["profile_name"] == "sparse_support_boost"
