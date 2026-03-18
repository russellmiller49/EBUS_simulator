from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ebus_simulator.manifest import _resolve_manifest_root
from ebus_simulator.models import CaseManifest


REFERENCE_MODALITIES = {"still_image", "clip_frame"}
REFERENCE_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_REFERENCE_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass(slots=True, frozen=True)
class ReferenceEntry:
    reference_id: str
    preset_id: str
    approach: str | None
    station: str
    image_path: Path
    modality: str
    correlation_confidence: str
    notes: str | None = None
    tags: tuple[str, ...] = ()
    vessel_visible: bool | None = None
    airway_wall_visible: bool | None = None
    node_prominent: bool | None = None


@dataclass(slots=True)
class ReferenceManifest:
    manifest_path: Path
    root: Path
    notes: dict[str, Any]
    entries: list[ReferenceEntry] = field(default_factory=list)


def _coerce_optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _coerce_optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Reference manifest field {key!r} must be a boolean when present.")


def _coerce_tags(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_tags = payload.get("tags", [])
    if raw_tags is None:
        return ()
    if not isinstance(raw_tags, list):
        raise ValueError("Reference manifest field 'tags' must be a list when present.")
    return tuple(str(item) for item in raw_tags)


def _validate_entry_against_case(entry: ReferenceEntry, case_manifest: CaseManifest) -> None:
    preset_lookup = {preset.id: preset for preset in case_manifest.presets}
    preset = preset_lookup.get(entry.preset_id)
    if preset is None:
        raise ValueError(
            f"Reference entry {entry.reference_id!r} points at unknown preset_id {entry.preset_id!r}."
        )
    if entry.approach is not None and entry.approach not in preset.contacts:
        raise ValueError(
            f"Reference entry {entry.reference_id!r} points at unsupported approach "
            f"{entry.approach!r} for preset {entry.preset_id!r}."
        )
    if entry.station.strip().lower() != preset.station.strip().lower():
        raise ValueError(
            f"Reference entry {entry.reference_id!r} declares station {entry.station!r}, "
            f"but preset {entry.preset_id!r} is station {preset.station!r}."
        )


def _parse_reference_entry(root: Path, payload: dict[str, Any], *, case_manifest: CaseManifest | None) -> ReferenceEntry:
    reference_id = str(payload["reference_id"])
    preset_id = str(payload["preset_id"])
    approach = _coerce_optional_string(payload, "approach")
    station = str(payload["station"])
    image_relative = Path(str(payload["image_path"]))
    image_path = (root / image_relative).resolve() if not image_relative.is_absolute() else image_relative.resolve()
    if not image_path.exists():
        raise FileNotFoundError(
            f"Reference image {image_path!s} does not exist for reference_id {reference_id!r}."
        )

    modality = str(payload["modality"])
    if modality not in REFERENCE_MODALITIES:
        raise ValueError(
            f"Reference entry {reference_id!r} has unsupported modality {modality!r}. "
            f"Expected one of: {', '.join(sorted(REFERENCE_MODALITIES))}."
        )

    correlation_confidence = str(payload["correlation_confidence"]).lower()
    if correlation_confidence not in REFERENCE_CONFIDENCE_LEVELS:
        raise ValueError(
            f"Reference entry {reference_id!r} has unsupported correlation_confidence "
            f"{correlation_confidence!r}. Expected one of: {', '.join(sorted(REFERENCE_CONFIDENCE_LEVELS))}."
        )

    entry = ReferenceEntry(
        reference_id=reference_id,
        preset_id=preset_id,
        approach=approach,
        station=station,
        image_path=image_path,
        modality=modality,
        correlation_confidence=correlation_confidence,
        notes=_coerce_optional_string(payload, "notes"),
        tags=_coerce_tags(payload),
        vessel_visible=_coerce_optional_bool(payload, "vessel_visible"),
        airway_wall_visible=_coerce_optional_bool(payload, "airway_wall_visible"),
        node_prominent=_coerce_optional_bool(payload, "node_prominent"),
    )
    if case_manifest is not None:
        _validate_entry_against_case(entry, case_manifest)
    return entry


def load_reference_manifest(
    path: str | Path,
    *,
    case_manifest: CaseManifest | None = None,
) -> ReferenceManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(manifest_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Reference manifest must be a YAML object.")

    root_value = payload.get("root")
    root = _resolve_manifest_root(manifest_path, root_value)
    if root is None:
        raise ValueError("Reference manifest root is required.")
    if not root.exists():
        raise FileNotFoundError(f"Reference manifest root {root!s} does not exist.")

    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("Reference manifest field 'entries' must be a list.")

    entries: list[ReferenceEntry] = []
    seen_reference_ids: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Reference manifest entries must be objects.")
        entry = _parse_reference_entry(root, raw_entry, case_manifest=case_manifest)
        if entry.reference_id in seen_reference_ids:
            raise ValueError(f"Duplicate reference_id {entry.reference_id!r} in reference manifest.")
        seen_reference_ids.add(entry.reference_id)
        entries.append(entry)

    entries.sort(
        key=lambda item: (
            item.preset_id,
            "" if item.approach is None else item.approach,
            _REFERENCE_CONFIDENCE_RANK[item.correlation_confidence],
            item.reference_id,
        )
    )
    return ReferenceManifest(
        manifest_path=manifest_path,
        root=root,
        notes=(payload.get("notes", {}) if isinstance(payload.get("notes", {}), dict) else {"summary": payload.get("notes")}),
        entries=entries,
    )


def find_reference_entries(
    reference_manifest: ReferenceManifest,
    *,
    preset_id: str,
    approach: str,
) -> list[ReferenceEntry]:
    matches = [
        entry
        for entry in reference_manifest.entries
        if entry.preset_id == preset_id and (entry.approach is None or entry.approach == approach)
    ]
    matches.sort(
        key=lambda entry: (
            0 if entry.approach == approach else 1,
            _REFERENCE_CONFIDENCE_RANK[entry.correlation_confidence],
            entry.reference_id,
        )
    )
    return matches


def reference_entry_to_dict(entry: ReferenceEntry) -> dict[str, object]:
    return {
        "reference_id": entry.reference_id,
        "preset_id": entry.preset_id,
        "approach": entry.approach,
        "station": entry.station,
        "image_path": str(entry.image_path),
        "modality": entry.modality,
        "correlation_confidence": entry.correlation_confidence,
        "notes": entry.notes,
        "tags": list(entry.tags),
        "vessel_visible": entry.vessel_visible,
        "airway_wall_visible": entry.airway_wall_visible,
        "node_prominent": entry.node_prominent,
    }
