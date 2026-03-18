from __future__ import annotations

from pathlib import Path

import pytest

from ebus_simulator.manifest import load_case_manifest
from ebus_simulator.reference_manifest import find_reference_entries, load_reference_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"
REFERENCE_MANIFEST_PATH = REPO_ROOT / "reference" / "manifest.yaml"


def test_reference_manifest_loads_repo_sample_entry():
    case_manifest = load_case_manifest(CASE_MANIFEST_PATH)

    reference_manifest = load_reference_manifest(REFERENCE_MANIFEST_PATH, case_manifest=case_manifest)

    assert reference_manifest.root == REPO_ROOT / "reference"
    assert len(reference_manifest.entries) == 1
    entry = reference_manifest.entries[0]
    assert entry.reference_id == "example_station_4r_node_b_placeholder"
    assert entry.preset_id == "station_4r_node_b"
    assert entry.approach == "default"
    assert entry.station == "4R"
    assert entry.modality == "still_image"
    assert entry.correlation_confidence == "low"
    assert entry.image_path.exists()
    assert "placeholder" in (entry.notes or "").lower()


def test_reference_manifest_rejects_duplicate_reference_ids(tmp_path):
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"root: {tmp_path}",
                "entries:",
                "  - reference_id: duplicate_ref",
                "    preset_id: station_4r_node_b",
                "    approach: default",
                "    station: 4R",
                "    image_path: example.png",
                "    modality: still_image",
                "    correlation_confidence: low",
                "  - reference_id: duplicate_ref",
                "    preset_id: station_7_node_a",
                "    station: 7",
                "    image_path: example.png",
                "    modality: still_image",
                "    correlation_confidence: medium",
                "",
            ]
        )
    )

    with pytest.raises(ValueError, match="Duplicate reference_id"):
        load_reference_manifest(manifest_path)


def test_find_reference_entries_matches_preset_wide_and_approach_specific_links(tmp_path):
    image_path = tmp_path / "example.png"
    image_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"root: {tmp_path}",
                "entries:",
                "  - reference_id: preset_wide_ref",
                "    preset_id: station_7_node_a",
                "    station: 7",
                "    image_path: example.png",
                "    modality: still_image",
                "    correlation_confidence: medium",
                "  - reference_id: lms_only_ref",
                "    preset_id: station_7_node_a",
                "    approach: lms",
                "    station: 7",
                "    image_path: example.png",
                "    modality: still_image",
                "    correlation_confidence: high",
                "",
            ]
        )
    )

    reference_manifest = load_reference_manifest(manifest_path)

    lms_references = find_reference_entries(reference_manifest, preset_id="station_7_node_a", approach="lms")
    rms_references = find_reference_entries(reference_manifest, preset_id="station_7_node_a", approach="rms")

    assert [entry.reference_id for entry in lms_references] == ["lms_only_ref", "preset_wide_ref"]
    assert [entry.reference_id for entry in rms_references] == ["preset_wide_ref"]
