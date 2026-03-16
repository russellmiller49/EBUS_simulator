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


def _resolve_optional_mesh(root: Path, explicit_value: str | None, fallback_relative: str) -> Path | None:
    resolved = _resolve_path(root, explicit_value) if explicit_value is not None else None
    if resolved is not None:
        return resolved
    fallback = root / fallback_relative
    return fallback if fallback.exists() else None


def load_case_manifest(path: str | Path) -> CaseManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(manifest_path.read_text())

    root_value = payload["root"]
    root = _resolve_path(manifest_path.parent, root_value)
    if root is None:
        raise ValueError("Manifest root is required.")

    centerlines = payload.get("centerlines", {})
    airway = payload.get("airway", {})
    meshes = payload.get("meshes", {})

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
        airway_raw_mesh=_resolve_optional_mesh(
            root,
            airway.get("raw_endoluminal_mesh", meshes.get("raw_airway_endoluminal_mesh")),
            "meshes/airway_endoluminal_surface_raw.vtp",
        ),
        airway_display_mesh=_resolve_optional_mesh(
            root,
            airway.get("smoothed_display_mesh", meshes.get("smoothed_airway_display_mesh")),
            "meshes/airway_endoluminal_surface_smoothed.vtp",
        ),
        airway_cutaway_display_mesh=_resolve_optional_mesh(
            root,
            airway.get("cutaway_display_mesh", meshes.get("cutaway_display_mesh")),
            "meshes/airway_endoluminal_surface_smoothed.vtp",
        ),
        station_masks=station_masks,
        overlay_masks=overlay_masks,
        presets=presets,
        qa=payload.get("qa", {}),
        render_defaults=payload.get("render_defaults", {}),
        notes=payload.get("notes", {}),
    )
