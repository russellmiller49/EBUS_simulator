from pathlib import Path
import json
from argparse import Namespace

from PIL import Image

from ebus_simulator.review_cli import _build_review_thresholds, _parse_optional_threshold
from ebus_simulator.review import (
    DEFAULT_REVIEW_THRESHOLDS,
    ReviewThresholds,
    _flag_review_metrics,
    compare_review_bundle_files,
    compare_review_summaries,
    review_presets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


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


def test_flag_review_metrics_can_include_physics_eval_thresholds():
    flags = _flag_review_metrics(
        {
            "nUS_delta_deg_from_voxel_baseline": 0.5,
            "contact_delta_mm_from_voxel_baseline": 0.2,
            "target_in_sector": True,
            "station_overlap_fraction_in_fan": 0.02,
            "contact_refinement_ambiguity": False,
        },
        physics_eval_summary={
            "target": {"pixel_count": 12},
            "wall": {"pixel_count": 0},
            "vessel": {"pixel_count": 8},
            "target_contrast_vs_sector": -0.015,
            "wall_contrast_vs_sector": None,
            "vessel_contrast_vs_sector": 0.004,
        },
        thresholds=ReviewThresholds(
            target_contrast_vs_sector_min=0.0,
            vessel_contrast_vs_sector_max=-0.01,
            wall_contrast_vs_sector_min=0.02,
        ),
    )

    assert any("target contrast" in reason for reason in flags)
    assert any("vessel contrast" in reason for reason in flags)
    assert any("wall region missing from physics eval summary" == reason for reason in flags)


def test_default_wall_threshold_is_enabled_and_cli_can_disable_it():
    assert DEFAULT_REVIEW_THRESHOLDS.wall_contrast_vs_sector_min == 0.02
    assert _parse_optional_threshold("0.03") == 0.03
    assert _parse_optional_threshold("off") is None
    assert _parse_optional_threshold("disabled") is None

    default_thresholds = _build_review_thresholds(
        Namespace(
            warn_nus_delta_deg=None,
            warn_contact_delta_mm=None,
            warn_station_overlap_fraction=None,
            warn_min_target_contrast=None,
            warn_max_vessel_contrast=None,
        )
    )
    assert default_thresholds.wall_contrast_vs_sector_min == 0.02

    disabled_thresholds = _build_review_thresholds(
        Namespace(
            warn_nus_delta_deg=None,
            warn_contact_delta_mm=None,
            warn_station_overlap_fraction=None,
            warn_min_target_contrast=None,
            warn_max_vessel_contrast=None,
            warn_min_wall_contrast=None,
        )
    )
    assert disabled_thresholds.wall_contrast_vs_sector_min is None


def test_review_presets_generates_physics_aware_bundle(tmp_path):
    output_dir = tmp_path / "review_bundle"
    summary = review_presets(
        MANIFEST_PATH,
        output_dir=output_dir,
        width=64,
        height=64,
        preset_ids=["station_4r_node_b", "station_7_node_a"],
        include_physics_debug_maps=True,
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
    assert summary["thresholds"]["target_contrast_vs_sector_min"] == 0.0
    assert summary["thresholds"]["vessel_contrast_vs_sector_max"] == -0.01
    assert summary["thresholds"]["wall_contrast_vs_sector_min"] == 0.02
    assert all("geometry_flag_reasons" in entry for entry in summary["entries"])
    assert all("physics_flag_reasons" in entry for entry in summary["entries"])
    assert all("target_contrast_vs_sector" in entry["physics_eval_summary"] for entry in summary["entries"])
    assert any(entry["physics_eval_summary"]["wall"]["pixel_count"] > 0 for entry in summary["entries"])
    assert index_payload["review_count"] == 3
    assert "| station_7_node_a | lms |" in index_markdown
    assert "| station_7_node_a | rms |" in index_markdown


def test_review_presets_can_include_reference_assets(tmp_path):
    reference_image = Image.new("RGB", (96, 64), (90, 90, 90))
    reference_image_path = tmp_path / "station_4r_reference.png"
    reference_image.save(reference_image_path)
    reference_config = tmp_path / "video_references.yaml"
    reference_config.write_text(
        "\n".join(
            [
                "reference_version: 1",
                f"root: {tmp_path.as_posix()}",
                "defaults:",
                "  ebus_keyframes: 1",
                "  frame_size_px: 64",
                "videos:",
                "  - id: station_4r_reference",
                "    station: 4r",
                "    kind: ebus",
                f"    path: {reference_image_path.name}",
                "    preset_ids: [station_4r_node_b]",
            ]
        )
    )

    output_dir = tmp_path / "reference_review_bundle"
    summary = review_presets(
        MANIFEST_PATH,
        output_dir=output_dir,
        width=64,
        height=64,
        preset_ids=["station_4r_node_b"],
        include_reference=True,
        reference_config=reference_config,
    )

    entry = summary["entries"][0]
    sheet = Path(entry["review_sheet_md"]).read_text()

    assert summary["include_reference"] is True
    assert Path(summary["reference_library_json"]).exists()
    assert entry["reference_status"] == "ebus_reference"
    assert len(entry["reference_keyframes"]) == 1
    assert "## Reference Material" in sheet
    assert "station_4r_reference_kf00" in sheet


def test_compare_review_summaries_tracks_flag_transitions_and_contrasts():
    before_summary = {
        "case_id": "demo_case",
        "review_count": 2,
        "flagged_count": 1,
        "thresholds": {"wall_contrast_vs_sector_min": None},
        "physics_settings": {"speckle_strength": 0.2},
        "entries": [
            {
                "preset_id": "station_4r_node_b",
                "approach": "default",
                "flagged": True,
                "flag_reasons": ["contact refinement remained ambiguous"],
                "metrics": {
                    "target_in_sector": True,
                    "station_overlap_fraction_in_fan": 0.20,
                    "nUS_delta_deg_from_voxel_baseline": 6.7,
                    "contact_delta_mm_from_voxel_baseline": 0.23,
                    "contact_refinement_ambiguity": True,
                    "target_lateral_offset_mm": -5.1,
                },
                "physics_eval_summary": {
                    "target_contrast_vs_sector": -0.02,
                    "wall_contrast_vs_sector": 0.01,
                    "vessel_contrast_vs_sector": 0.03,
                    "wall": {"pixel_count": 4},
                },
            },
            {
                "preset_id": "station_7_node_a",
                "approach": "lms",
                "flagged": False,
                "flag_reasons": [],
                "metrics": {
                    "target_in_sector": True,
                    "station_overlap_fraction_in_fan": 0.70,
                    "nUS_delta_deg_from_voxel_baseline": 2.5,
                    "contact_delta_mm_from_voxel_baseline": 0.50,
                    "contact_refinement_ambiguity": False,
                    "target_lateral_offset_mm": -8.0,
                },
                "physics_eval_summary": {
                    "target_contrast_vs_sector": 0.05,
                    "wall_contrast_vs_sector": 0.04,
                    "vessel_contrast_vs_sector": -0.03,
                    "wall": {"pixel_count": 7},
                },
            },
        ],
    }
    after_summary = {
        "case_id": "demo_case",
        "review_count": 2,
        "flagged_count": 1,
        "thresholds": {"wall_contrast_vs_sector_min": 0.02},
        "physics_settings": {"speckle_strength": 0.22},
        "entries": [
            {
                "preset_id": "station_4r_node_b",
                "approach": "default",
                "flagged": False,
                "flag_reasons": [],
                "metrics": {
                    "target_in_sector": True,
                    "station_overlap_fraction_in_fan": 0.22,
                    "nUS_delta_deg_from_voxel_baseline": 3.8,
                    "contact_delta_mm_from_voxel_baseline": 0.04,
                    "contact_refinement_ambiguity": False,
                    "target_lateral_offset_mm": -0.3,
                },
                "physics_eval_summary": {
                    "target_contrast_vs_sector": 0.01,
                    "wall_contrast_vs_sector": 0.03,
                    "vessel_contrast_vs_sector": -0.02,
                    "wall": {"pixel_count": 9},
                },
            },
            {
                "preset_id": "station_7_node_a",
                "approach": "lms",
                "flagged": True,
                "flag_reasons": ["wall contrast 0.010 < 0.020"],
                "metrics": {
                    "target_in_sector": True,
                    "station_overlap_fraction_in_fan": 0.72,
                    "nUS_delta_deg_from_voxel_baseline": 2.4,
                    "contact_delta_mm_from_voxel_baseline": 0.49,
                    "contact_refinement_ambiguity": False,
                    "target_lateral_offset_mm": -7.7,
                },
                "physics_eval_summary": {
                    "target_contrast_vs_sector": 0.04,
                    "wall_contrast_vs_sector": 0.01,
                    "vessel_contrast_vs_sector": -0.04,
                    "wall": {"pixel_count": 8},
                },
            },
        ],
    }

    comparison = compare_review_summaries(before_summary, after_summary)

    assert comparison["case_id_match"] is True
    assert comparison["matched_entry_count"] == 2
    assert comparison["resolved_flagged_count"] == 1
    assert comparison["regressed_flagged_count"] == 1
    assert comparison["unchanged_flagged_count"] == 0
    assert comparison["unchanged_clear_count"] == 0
    assert comparison["before_thresholds"]["wall_contrast_vs_sector_min"] is None
    assert comparison["after_thresholds"]["wall_contrast_vs_sector_min"] == 0.02

    station_4r = next(row for row in comparison["rows"] if row["preset_id"] == "station_4r_node_b")
    assert station_4r["flag_transition"] == "resolved"
    assert station_4r["before_wall_pixel_count"] == 4
    assert station_4r["after_wall_pixel_count"] == 9
    assert station_4r["before_wall_contrast_vs_sector"] == 0.01
    assert station_4r["after_wall_contrast_vs_sector"] == 0.03

    station_7 = next(row for row in comparison["rows"] if row["preset_id"] == "station_7_node_a")
    assert station_7["flag_transition"] == "regressed"
    assert station_7["before_reasons"] == []
    assert station_7["after_reasons"] == ["wall contrast 0.010 < 0.020"]


def test_compare_review_bundle_files_writes_artifacts(tmp_path):
    before_summary = {
        "case_id": "demo_case",
        "review_count": 1,
        "flagged_count": 1,
        "entries": [
            {
                "preset_id": "station_4r_node_b",
                "approach": "default",
                "flagged": True,
                "flag_reasons": ["contact refinement remained ambiguous"],
                "metrics": {
                    "target_in_sector": True,
                    "station_overlap_fraction_in_fan": 0.20,
                    "nUS_delta_deg_from_voxel_baseline": 6.7,
                    "contact_delta_mm_from_voxel_baseline": 0.23,
                    "contact_refinement_ambiguity": True,
                    "target_lateral_offset_mm": -5.1,
                },
                "physics_eval_summary": {
                    "target_contrast_vs_sector": -0.02,
                    "wall_contrast_vs_sector": 0.01,
                    "vessel_contrast_vs_sector": 0.03,
                    "wall": {"pixel_count": 4},
                },
            }
        ],
    }
    after_summary = {
        "case_id": "demo_case",
        "review_count": 1,
        "flagged_count": 0,
        "entries": [
            {
                "preset_id": "station_4r_node_b",
                "approach": "default",
                "flagged": False,
                "flag_reasons": [],
                "metrics": {
                    "target_in_sector": True,
                    "station_overlap_fraction_in_fan": 0.22,
                    "nUS_delta_deg_from_voxel_baseline": 3.8,
                    "contact_delta_mm_from_voxel_baseline": 0.04,
                    "contact_refinement_ambiguity": False,
                    "target_lateral_offset_mm": -0.3,
                },
                "physics_eval_summary": {
                    "target_contrast_vs_sector": 0.01,
                    "wall_contrast_vs_sector": 0.03,
                    "vessel_contrast_vs_sector": -0.02,
                    "wall": {"pixel_count": 9},
                },
            }
        ],
    }

    before_path = tmp_path / "before_review_summary.json"
    after_path = tmp_path / "after_review_summary.json"
    output_dir = tmp_path / "comparison"
    before_path.write_text(json.dumps(before_summary, indent=2))
    after_path.write_text(json.dumps(after_summary, indent=2))

    comparison = compare_review_bundle_files(before_path, after_path, output_dir=output_dir)

    assert comparison["resolved_flagged_count"] == 1
    assert Path(comparison["comparison_json"]).exists()
    assert Path(comparison["comparison_csv"]).exists()
    assert Path(comparison["comparison_md"]).exists()

    markdown = Path(comparison["comparison_md"]).read_text()
    assert "Resolved Flags" in markdown
    assert "`station_4r_node_b` / `default`" in markdown

    csv_text = Path(comparison["comparison_csv"]).read_text()
    assert "flag_transition" in csv_text
    assert "resolved" in csv_text
