from pathlib import Path
import json

import numpy as np
from PIL import Image

from ebus_simulator.reference_annotations import load_cvat_coco_annotations, normalize_reference_label
from ebus_simulator.reference_metrics import summarize_reference_keyframe
from ebus_simulator.video_reference import (
    build_reference_library,
    load_video_reference_config,
    station_reference_keyframes,
    station_reference_status,
)


def _write_reference_config(tmp_path: Path, image_path: Path) -> Path:
    config_path = tmp_path / "video_references.yaml"
    config_path.write_text(
        "\n".join(
            [
                "reference_version: 1",
                f"root: {tmp_path.as_posix()}",
                "defaults:",
                "  ebus_keyframes: 5",
                "  white_light_keyframes: 3",
                "  frame_size_px: 80",
                "videos:",
                "  - id: station_4r_fixture",
                "    station: 4r",
                "    kind: ebus",
                f"    path: {image_path.name}",
                "    preset_ids: [station_4r_node_b]",
                "    deidentify_regions:",
                "      - {x: 0.0, y: 0.0, width: 0.25, height: 0.25}",
            ]
        )
    )
    return config_path


def test_video_reference_config_and_library_builder_are_deterministic(tmp_path):
    image = Image.new("RGB", (120, 80), (120, 120, 120))
    image_path = tmp_path / "reference.png"
    image.save(image_path)
    config_path = _write_reference_config(tmp_path, image_path)

    config = load_video_reference_config(config_path)
    library = build_reference_library(config_path, output_dir=tmp_path / "library")

    assert config.videos[0].id == "station_4r_fixture"
    assert config.videos[0].stations == ["4r"]
    assert len(library.keyframes) == 1
    assert station_reference_status(library, "4r") == "ebus_reference"
    assert station_reference_keyframes(library, "4r")[0]["id"] == "station_4r_fixture_kf00"

    keyframe_path = Path(library.keyframes[0]["image_path"])
    keyframe = np.asarray(Image.open(keyframe_path))
    assert keyframe.shape[0] <= 80
    assert np.all(keyframe[:5, :5, :] == 0)


def test_cvat_coco_annotations_feed_reference_metrics(tmp_path):
    image = Image.new("L", (64, 64), 80)
    image_path = tmp_path / "station_4r_fixture_kf00.png"
    image.save(image_path)
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image_path.name, "width": 64, "height": 64}],
                "categories": [
                    {"id": 1, "name": "ultrasound_fan"},
                    {"id": 2, "name": "lymph_node"},
                    {"id": 3, "name": "vessel_lumen"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "segmentation": [[0, 0, 63, 0, 63, 63, 0, 63]], "area": 4096},
                    {"image_id": 1, "category_id": 2, "segmentation": [[16, 16, 32, 16, 32, 32, 16, 32]], "area": 256},
                    {"image_id": 1, "category_id": 3, "segmentation": [[40, 16, 52, 16, 52, 28, 40, 28]], "area": 144},
                ],
            }
        )
    )

    summaries = load_cvat_coco_annotations(coco_path)
    metrics = summarize_reference_keyframe(image_path, summaries[image_path.name], keyframe_id="station_4r_fixture_kf00")

    assert summaries[image_path.name].label_counts["lymph_node"] == 1
    assert metrics.keyframe_id == "station_4r_fixture_kf00"
    assert metrics.fan_area_fraction is not None
    assert metrics.label_pixel_counts["lymph_node"] > 0
    assert metrics.speckle_std_in_fan == 0.0


def test_annotation_label_aliases_match_cvat_export_names():
    assert normalize_reference_label("US_field") == "ultrasound_fan"
    assert normalize_reference_label("Lymph node") == "lymph_node"
    assert normalize_reference_label("Pulmonary Artery") == "vessel_lumen"
    assert normalize_reference_label("Station 11Ri") == "lymph_node"
