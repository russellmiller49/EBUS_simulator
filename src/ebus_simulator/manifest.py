from __future__ import annotations

from pathlib import Path

import yaml

from ebus_simulator.models import CaseManifest, ManifestPreset, ManifestPresetOverrides


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


def _parse_preset_overrides(payload: dict | None) -> ManifestPresetOverrides | None:
    if not payload:
        return None
    return ManifestPresetOverrides(
        vessel_overlays=(None if payload.get("vessel_overlays") is None else [str(item) for item in payload.get("vessel_overlays", [])]),
        cutaway_side=(None if payload.get("cutaway_side") is None else str(payload.get("cutaway_side"))),
        roll_offset_deg=(None if payload.get("roll_offset_deg") is None else float(payload.get("roll_offset_deg"))),
        branch_hint=(None if payload.get("branch_hint") is None else str(payload.get("branch_hint"))),
        axis_sign_override=(None if payload.get("axis_sign_override") is None else str(payload.get("axis_sign_override"))),
        reference_fov_mm=(None if payload.get("reference_fov_mm") is None else float(payload.get("reference_fov_mm"))),
        notes=(None if payload.get("notes") is None else str(payload.get("notes"))),
    )


def merge_preset_overrides(
    base: ManifestPresetOverrides | None,
    approach: ManifestPresetOverrides | None,
) -> ManifestPresetOverrides | None:
    if base is None and approach is None:
        return None
    return ManifestPresetOverrides(
        vessel_overlays=(approach.vessel_overlays if approach is not None and approach.vessel_overlays is not None else None if base is None else base.vessel_overlays),
        cutaway_side=(approach.cutaway_side if approach is not None and approach.cutaway_side is not None else None if base is None else base.cutaway_side),
        roll_offset_deg=(approach.roll_offset_deg if approach is not None and approach.roll_offset_deg is not None else None if base is None else base.roll_offset_deg),
        branch_hint=(approach.branch_hint if approach is not None and approach.branch_hint is not None else None if base is None else base.branch_hint),
        axis_sign_override=(approach.axis_sign_override if approach is not None and approach.axis_sign_override is not None else None if base is None else base.axis_sign_override),
        reference_fov_mm=(
            approach.reference_fov_mm
            if approach is not None and approach.reference_fov_mm is not None
            else None if base is None else base.reference_fov_mm
        ),
        notes=(approach.notes if approach is not None and approach.notes is not None else None if base is None else base.notes),
    )


def resolve_preset_overrides(
    preset: ManifestPreset,
    *,
    approach: str | None,
) -> ManifestPresetOverrides | None:
    if approach is None:
        return preset.overrides
    return merge_preset_overrides(preset.overrides, preset.approach_overrides.get(approach))


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
                overrides=_parse_preset_overrides(preset.get("overrides")),
                approach_overrides={
                    str(name): parsed_override
                    for name, parsed_override in (
                        (
                            str(name),
                            _parse_preset_overrides(override_payload),
                        )
                        for name, override_payload in preset.get("approach_overrides", {}).items()
                    )
                    if parsed_override is not None
                },
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
