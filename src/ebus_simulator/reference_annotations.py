from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REFERENCE_LABELS = {
    "ultrasound_fan",
    "depth_marker",
    "lymph_node",
    "vessel_lumen",
    "airway_wall_interface",
    "reverberation",
    "shadow",
    "white_light_lumen",
    "branch_landmark",
    "contact_region",
}


REFERENCE_LABEL_ALIASES = {
    "us_field": "ultrasound_fan",
    "us field": "ultrasound_fan",
    "ultrasound field": "ultrasound_fan",
    "lymph node": "lymph_node",
    "node": "lymph_node",
    "vessel": "vessel_lumen",
    "vessel lumen": "vessel_lumen",
    "aorta": "vessel_lumen",
    "pulmonary artery": "vessel_lumen",
    "brachiocephalic trunk": "vessel_lumen",
    "azygous vein": "vessel_lumen",
    "brachiocephalic vein": "vessel_lumen",
    "superior vena cava": "vessel_lumen",
    "left atrium": "vessel_lumen",
    "left common carotid": "vessel_lumen",
    "left common carotid artery": "vessel_lumen",
    "station 2r": "lymph_node",
    "station 2l": "lymph_node",
    "station 4r": "lymph_node",
    "station 4l": "lymph_node",
    "station 5": "lymph_node",
    "station 7": "lymph_node",
    "station 10r": "lymph_node",
    "station 10l": "lymph_node",
    "station 11rs": "lymph_node",
    "station 11ri": "lymph_node",
    "station 11l": "lymph_node",
}


@dataclass(slots=True)
class AnnotationRegion:
    label: str
    segmentation: list[list[float]]
    area: float | None = None
    bbox: list[float] | None = None
    source_label: str | None = None


@dataclass(slots=True)
class ReferenceAnnotationSummary:
    image_id: int
    file_name: str
    width: int
    height: int
    regions: list[AnnotationRegion] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    label_areas_px: dict[str, float] = field(default_factory=dict)
    unknown_labels: list[str] = field(default_factory=list)


def normalize_reference_label(label: str) -> str:
    normalized = str(label).strip().lower().replace("_", " ")
    return REFERENCE_LABEL_ALIASES.get(normalized, normalized.replace(" ", "_"))


def load_cvat_coco_annotations(path: str | Path) -> dict[str, ReferenceAnnotationSummary]:
    annotation_path = Path(path).expanduser().resolve()
    payload = json.loads(annotation_path.read_text())

    categories = {
        int(category["id"]): str(category["name"])
        for category in payload.get("categories", [])
    }
    summaries: dict[int, ReferenceAnnotationSummary] = {}
    for image in payload.get("images", []):
        image_id = int(image["id"])
        summaries[image_id] = ReferenceAnnotationSummary(
            image_id=image_id,
            file_name=str(image["file_name"]),
            width=int(image.get("width", 0)),
            height=int(image.get("height", 0)),
        )

    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in summaries:
            continue
        source_label = categories.get(int(annotation["category_id"]), f"unknown_{annotation['category_id']}")
        label = normalize_reference_label(source_label)
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            segmentation = []
        polygons = [
            [float(value) for value in polygon]
            for polygon in segmentation
            if isinstance(polygon, list) and len(polygon) >= 6
        ]
        summary = summaries[image_id]
        summary.regions.append(
            AnnotationRegion(
                label=label,
                segmentation=polygons,
                area=(None if annotation.get("area") is None else float(annotation["area"])),
                bbox=(None if annotation.get("bbox") is None else [float(value) for value in annotation["bbox"]]),
                source_label=source_label,
            )
        )
        summary.label_counts[label] = summary.label_counts.get(label, 0) + 1
        summary.label_areas_px[label] = summary.label_areas_px.get(label, 0.0) + float(annotation.get("area") or 0.0)
        if label not in REFERENCE_LABELS and label not in summary.unknown_labels:
            summary.unknown_labels.append(label)

    return {summary.file_name: summary for summary in summaries.values()}


def annotation_masks(summary: ReferenceAnnotationSummary) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for region in summary.regions:
        mask = masks.setdefault(region.label, np.zeros((summary.height, summary.width), dtype=bool))
        image = Image.new("L", (summary.width, summary.height), 0)
        draw = ImageDraw.Draw(image)
        for polygon in region.segmentation:
            xy = [(polygon[index], polygon[index + 1]) for index in range(0, len(polygon) - 1, 2)]
            if len(xy) >= 3:
                draw.polygon(xy, outline=1, fill=1)
        mask |= np.asarray(image, dtype=np.uint8) > 0
    return masks


def write_annotation_summary(path: str | Path, summaries: dict[str, ReferenceAnnotationSummary]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        file_name: asdict(summary)
        for file_name, summary in summaries.items()
    }
    output_path.write_text(json.dumps(payload, indent=2))


def render_annotation_overlay(
    image_path: str | Path,
    summary: ReferenceAnnotationSummary,
    output_path: str | Path,
) -> None:
    colors = {
        "ultrasound_fan": (255, 255, 0),
        "lymph_node": (0, 255, 255),
        "vessel_lumen": (0, 128, 255),
        "airway_wall_interface": (255, 160, 0),
        "reverberation": (255, 0, 255),
        "shadow": (128, 128, 255),
        "white_light_lumen": (0, 255, 128),
        "branch_landmark": (255, 255, 255),
        "contact_region": (255, 64, 64),
    }
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in summary.regions:
        color = colors.get(region.label, (255, 255, 255))
        for polygon in region.segmentation:
            xy = [(polygon[index], polygon[index + 1]) for index in range(0, len(polygon) - 1, 2)]
            if len(xy) >= 3:
                draw.line(xy + [xy[0]], fill=color, width=2)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
