from pathlib import Path

import pytest

from ebus_simulator.manifest import load_case_manifest


def _write_manifest(tmp_path: Path, *, root_value: str) -> tuple[Path, Path]:
    repo_root = tmp_path
    (repo_root / ".git").mkdir()
    (repo_root / "configs").mkdir()
    dataset_root = repo_root / "dataset"
    dataset_root.mkdir()
    manifest_path = repo_root / "configs" / "case.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "manifest_version: 1",
                "case_id: test_case",
                f'root: "{root_value}"',
                "ct:",
                '  image: "ct.nii.gz"',
                "centerlines:",
                '  main: "centerlines/airway_centerline.vtp"',
                '  network: "centerlines/airway_network.vtp"',
                "airway:",
                '  lumen_mask: "masks/airway.nii.gz"',
                '  solid_mask: "masks/airway_solid.nii.gz"',
                "station_masks: {}",
                "overlay_masks: {}",
                "presets: []",
            ]
        )
    )
    return manifest_path, dataset_root


def test_manifest_root_supports_repo_relative_path(tmp_path):
    manifest_path, dataset_root = _write_manifest(tmp_path, root_value="dataset")
    manifest = load_case_manifest(manifest_path)
    assert manifest.root == dataset_root.resolve()


def test_manifest_root_supports_repo_root_token(tmp_path):
    manifest_path, dataset_root = _write_manifest(tmp_path, root_value="${REPO_ROOT}/dataset")
    manifest = load_case_manifest(manifest_path)
    assert manifest.root == dataset_root.resolve()


def test_manifest_root_supports_data_root_token(tmp_path, monkeypatch):
    manifest_path, _ = _write_manifest(tmp_path, root_value="${DATA_ROOT}")
    external_root = tmp_path / "external_dataset"
    external_root.mkdir()
    monkeypatch.setenv("DATA_ROOT", str(external_root))
    manifest = load_case_manifest(manifest_path)
    assert manifest.root == external_root.resolve()


def test_manifest_root_missing_raises_clear_error(tmp_path):
    manifest_path, _ = _write_manifest(tmp_path, root_value="${REPO_ROOT}/missing_dataset")
    with pytest.raises(FileNotFoundError, match="Original root value"):
        load_case_manifest(manifest_path)
