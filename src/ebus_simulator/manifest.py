from __future__ import annotations

from pathlib import Path

import yaml

from ebus_simulator.models import CaseManifest, ManifestPreset


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_case_manifest(path: str | Path) -> CaseManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(manifest_path.read_text())

    root_value = payload["root"]
    root = _resolve_path(manifest_path.parent, root_value)
    if root is None:
        raise ValueError("Manifest root is required.")

    centerlines = payload.get("centerlines", {})
    airway = payload.get("airway", {})

    presets: list[ManifestPreset] = []
    for preset in payload.get("presets", []):
        contacts = {
            name: _resolve_path(root, contact_path)
            for name, contact_path in preset.get("contacts", {}).items()
        }
        presets.append(
            ManifestPreset(
                id=preset["id"],
                station=str(preset["station"]),
                node=str(preset["node"]),
                station_mask=_resolve_path(root, preset["station_mask"]),
                target=_resolve_path(root, preset["target"]),
                contacts=contacts,
            )
        )

    station_masks = {
        key: _resolve_path(root, value)
        for key, value in payload.get("station_masks", {}).items()
    }
    overlay_masks = {
        key: _resolve_path(root, value)
        for key, value in payload.get("overlay_masks", {}).items()
    }

    return CaseManifest(
        manifest_path=manifest_path,
        case_id=payload["case_id"],
        root=root,
        ct_image=_resolve_path(root, payload["ct"]["image"]),
        centerline_main=_resolve_path(root, centerlines["main"]),
        centerline_network=_resolve_path(root, centerlines["network"]),
        primary_markup_curve=_resolve_path(root, centerlines.get("primary_markup_curve")),
        secondary_network_curves=[
            _resolve_path(root, item)
            for item in centerlines.get("secondary_network_curves", [])
        ],
        airway_lumen_mask=_resolve_path(root, airway["lumen_mask"]),
        airway_solid_mask=_resolve_path(root, airway["solid_mask"]),
        station_masks=station_masks,
        overlay_masks=overlay_masks,
        presets=presets,
        qa=payload.get("qa", {}),
        render_defaults=payload.get("render_defaults", {}),
        notes=payload.get("notes", {}),
    )
