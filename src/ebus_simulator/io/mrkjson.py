from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ebus_simulator.models import MarkupControlPoint, MarkupFile, MarkupNode

RAS_TO_LPS_3X3 = np.diag([-1.0, -1.0, 1.0])


def _normalize_coordinate_system(raw: str | int | None) -> str:
    mapping = {
        "0": "RAS",
        "1": "LPS",
        0: "RAS",
        1: "LPS",
        "RAS": "RAS",
        "LPS": "LPS",
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported markup coordinate system: {raw!r}")
    return mapping[raw]


def _to_lps(position: list[float], coordinate_system: str) -> np.ndarray:
    point = np.asarray(position, dtype=np.float64)
    if coordinate_system == "LPS":
        return point
    if coordinate_system == "RAS":
        return RAS_TO_LPS_3X3 @ point
    raise ValueError(f"Unsupported coordinate system: {coordinate_system}")


def load_mrk_json(path: str | Path) -> MarkupFile:
    resolved_path = Path(path).expanduser().resolve()
    payload = json.loads(resolved_path.read_text())
    markup_nodes: list[MarkupNode] = []

    for markup in payload.get("markups", []):
        coordinate_system = _normalize_coordinate_system(markup.get("coordinateSystem"))
        control_points: list[MarkupControlPoint] = []
        for point in markup.get("controlPoints", []):
            raw_position = np.asarray(point["position"], dtype=np.float64)
            control_points.append(
                MarkupControlPoint(
                    id=str(point.get("id", "")),
                    label=str(point.get("label", "")),
                    position_lps=_to_lps(point["position"], coordinate_system),
                    position_raw=raw_position,
                    position_status=str(point.get("positionStatus", "")),
                )
            )
        markup_nodes.append(
            MarkupNode(
                type=str(markup.get("type", "")),
                coordinate_system=coordinate_system,
                coordinate_units=markup.get("coordinateUnits"),
                control_points=control_points,
            )
        )

    return MarkupFile(path=resolved_path, markups=markup_nodes)


def get_first_defined_control_point(markup_file: MarkupFile) -> MarkupControlPoint:
    for node in markup_file.markups:
        for point in node.control_points:
            if point.position_status == "defined":
                return point
    raise ValueError(f"No defined control point found in {markup_file.path}")


def load_first_defined_control_point(path: str | Path) -> MarkupControlPoint:
    return get_first_defined_control_point(load_mrk_json(path))
