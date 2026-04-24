from __future__ import annotations

import argparse
from csv import DictWriter
from dataclasses import asdict
import json
from pathlib import Path
import re

from ebus_simulator.reference_annotations import (
    ReferenceAnnotationSummary,
    load_cvat_coco_annotations,
    render_annotation_overlay,
    write_annotation_summary,
)
from ebus_simulator.reference_metrics import (
    comparison_metrics_to_dict,
    summarize_reference_keyframe,
)


STATION_PATTERNS = (
    ("11rs", re.compile(r"11\\s*rs|11rs", re.IGNORECASE)),
    ("11ri", re.compile(r"11\\s*ri|11ri", re.IGNORECASE)),
    ("11l", re.compile(r"11\\s*l|11l", re.IGNORECASE)),
    ("10r", re.compile(r"10\\s*r|10r", re.IGNORECASE)),
    ("10l", re.compile(r"10\\s*l|10l", re.IGNORECASE)),
    ("4r", re.compile(r"4\\s*r|4r", re.IGNORECASE)),
    ("4l", re.compile(r"4\\s*l|4l", re.IGNORECASE)),
    ("2r", re.compile(r"2\\s*r|2r", re.IGNORECASE)),
    ("2l", re.compile(r"2\\s*l|2l", re.IGNORECASE)),
    ("7", re.compile(r"station[_\\s-]*7|\\b7\\b", re.IGNORECASE)),
    ("5", re.compile(r"station[_\\s-]*5|\\b5\\b", re.IGNORECASE)),
)


def _station_from_text(text: str) -> str | None:
    for station, pattern in STATION_PATTERNS:
        if pattern.search(text):
            return station
    return None


def infer_station(summary: ReferenceAnnotationSummary) -> str | None:
    for region in summary.regions:
        if region.source_label is not None:
            station = _station_from_text(region.source_label)
            if station is not None:
                return station
    return _station_from_text(summary.file_name)


def _format_optional(value: object, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{precision}f}"


def summarize_annotation_folder(
    image_dir: str | Path,
    annotations_json: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    image_dir = Path(image_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_cvat_coco_annotations(annotations_json)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for file_name, summary in sorted(summaries.items()):
        image_path = image_dir / file_name
        if not image_path.exists():
            warnings.append(f"Missing image for annotation entry: {file_name}")
            continue
        overlay_path = overlay_dir / f"{Path(file_name).stem}_overlay.png"
        render_annotation_overlay(image_path, summary, overlay_path)
        metrics = summarize_reference_keyframe(image_path, summary, keyframe_id=Path(file_name).stem)
        metrics_payload = comparison_metrics_to_dict(metrics)
        rows.append(
            {
                "file_name": file_name,
                "station": infer_station(summary),
                "overlay_png": str(overlay_path),
                "label_counts": dict(summary.label_counts),
                "unknown_labels": list(summary.unknown_labels),
                **metrics_payload,
            }
        )

    station_counts: dict[str, int] = {}
    for row in rows:
        station = str(row.get("station") or "unknown")
        station_counts[station] = station_counts.get(station, 0) + 1

    payload = {
        "image_dir": str(image_dir),
        "annotations_json": str(Path(annotations_json).expanduser().resolve()),
        "output_dir": str(output_dir),
        "image_count": len(summaries),
        "processed_image_count": len(rows),
        "station_counts": station_counts,
        "warnings": warnings,
        "rows": rows,
    }

    write_annotation_summary(output_dir / "annotation_summary.json", summaries)
    (output_dir / "annotation_metrics.json").write_text(json.dumps(payload, indent=2))
    _write_metrics_csv(output_dir / "annotation_metrics.csv", rows)
    _write_metrics_markdown(output_dir / "annotation_metrics.md", payload)
    return payload


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "file_name",
        "station",
        "fan_area_fraction",
        "lymph_node_contrast_vs_fan",
        "vessel_contrast_vs_fan",
        "wall_contrast_vs_fan",
        "shadow_contrast_vs_fan",
        "near_field_wall_mean",
        "speckle_std_in_fan",
        "label_counts",
        "unknown_labels",
        "overlay_png",
    ]
    with path.open("w", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if key in {"label_counts", "unknown_labels"}
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _write_metrics_markdown(path: Path, payload: dict[str, object]) -> None:
    rows = payload["rows"]
    lines = [
        "# EBUS Annotation Metrics",
        "",
        f"- processed images: {payload['processed_image_count']} / {payload['image_count']}",
        f"- station counts: {payload['station_counts']}",
        "",
        "| Image | Station | Labels | Fan Area | Node Contrast | Vessel Contrast | Speckle Std | Overlay |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        labels = ", ".join(f"{key}:{value}" for key, value in sorted(row["label_counts"].items()))
        lines.append(
            "| {image} | {station} | {labels} | {fan} | {node} | {vessel} | {speckle} | [overlay]({overlay}) |".format(
                image=row["file_name"],
                station=row.get("station") or "unknown",
                labels=labels or "none",
                fan=_format_optional(row.get("fan_area_fraction")),
                node=_format_optional(row.get("lymph_node_contrast_vs_fan")),
                vessel=_format_optional(row.get("vessel_contrast_vs_fan")),
                speckle=_format_optional(row.get("speckle_std_in_fan")),
                overlay=Path(str(row["overlay_png"])).relative_to(path.parent),
            )
        )
    if payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CVAT/COCO EBUS reference-image annotations.")
    parser.add_argument("image_dir", help="Directory containing the annotated images.")
    parser.add_argument("annotations_json", help="COCO annotations JSON exported from CVAT.")
    parser.add_argument("--output-dir", required=True, help="Directory for overlays and metric summaries.")
    args = parser.parse_args()

    payload = summarize_annotation_folder(args.image_dir, args.annotations_json, output_dir=args.output_dir)
    print(f"image_count: {payload['image_count']}")
    print(f"processed_image_count: {payload['processed_image_count']}")
    print(f"station_counts: {payload['station_counts']}")
    print(f"warnings: {len(payload['warnings'])}")
    print(f"metrics_json: {payload['output_dir']}/annotation_metrics.json")
    print(f"metrics_csv: {payload['output_dir']}/annotation_metrics.csv")
    print(f"metrics_md: {payload['output_dir']}/annotation_metrics.md")
    print(f"overlays: {payload['output_dir']}/overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
