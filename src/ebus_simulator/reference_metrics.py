from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ebus_simulator.reference_annotations import ReferenceAnnotationSummary, annotation_masks


@dataclass(slots=True)
class ReferenceComparisonMetrics:
    keyframe_id: str | None
    image_path: str
    fan_area_fraction: float | None
    lymph_node_contrast_vs_fan: float | None
    vessel_contrast_vs_fan: float | None
    wall_contrast_vs_fan: float | None
    shadow_contrast_vs_fan: float | None
    near_field_wall_mean: float | None
    speckle_std_in_fan: float | None
    label_pixel_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _grayscale(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def _mean_or_none(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return None
    return float(np.mean(selected))


def _contrast(values: np.ndarray, mask: np.ndarray, baseline: float | None) -> float | None:
    region_mean = _mean_or_none(values, mask)
    if region_mean is None or baseline is None:
        return None
    return float(region_mean - baseline)


def summarize_reference_keyframe(
    image_path: str | Path,
    annotation_summary: ReferenceAnnotationSummary | None,
    *,
    keyframe_id: str | None = None,
) -> ReferenceComparisonMetrics:
    image = _grayscale(image_path)
    warnings: list[str] = []
    if annotation_summary is None:
        fan_mask = image > 0.02
        masks: dict[str, np.ndarray] = {}
        warnings.append("No annotation summary was provided; non-black pixels used as the fan mask.")
    else:
        masks = annotation_masks(annotation_summary)
        fan_mask = masks.get("ultrasound_fan")
        if fan_mask is None:
            fan_mask = image > 0.02
            warnings.append("No ultrasound_fan annotation was provided; non-black pixels used as the fan mask.")

    fan_mean = _mean_or_none(image, fan_mask)
    label_pixel_counts = {
        label: int(np.count_nonzero(mask))
        for label, mask in masks.items()
    }
    wall_mask = masks.get("airway_wall_interface", np.zeros_like(fan_mask, dtype=bool))
    near_field_limit = max(1, image.shape[0] // 4)
    near_field_wall = np.zeros_like(fan_mask, dtype=bool)
    near_field_wall[:near_field_limit, :] = wall_mask[:near_field_limit, :]

    return ReferenceComparisonMetrics(
        keyframe_id=keyframe_id,
        image_path=str(Path(image_path).expanduser().resolve()),
        fan_area_fraction=float(np.count_nonzero(fan_mask) / fan_mask.size) if fan_mask.size else None,
        lymph_node_contrast_vs_fan=_contrast(image, masks.get("lymph_node", np.zeros_like(fan_mask, dtype=bool)), fan_mean),
        vessel_contrast_vs_fan=_contrast(image, masks.get("vessel_lumen", np.zeros_like(fan_mask, dtype=bool)), fan_mean),
        wall_contrast_vs_fan=_contrast(image, wall_mask, fan_mean),
        shadow_contrast_vs_fan=_contrast(image, masks.get("shadow", np.zeros_like(fan_mask, dtype=bool)), fan_mean),
        near_field_wall_mean=_mean_or_none(image, near_field_wall),
        speckle_std_in_fan=(None if not np.any(fan_mask) else float(np.std(image[fan_mask]))),
        label_pixel_counts=label_pixel_counts,
        warnings=warnings,
    )


def comparison_metrics_to_dict(metrics: ReferenceComparisonMetrics) -> dict[str, object]:
    return asdict(metrics)
