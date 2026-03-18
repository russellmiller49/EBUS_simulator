from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import yaml


DEFAULT_RENDER_PROFILES_PATH = Path(__file__).resolve().parents[2] / "configs" / "render_profiles.yaml"
DEFAULT_PHYSICS_PROFILE_NAME = "baseline"
DEFAULT_CONSISTENCY_PROFILE_NAME = "baseline"


@dataclass(frozen=True, slots=True)
class ConsistencyTuningProfile:
    signal_threshold: float = 0.05
    near_field_fraction: float = 0.20
    target_region_radius_mm: float = 4.0
    target_prominent_min_contrast: float = 0.08
    target_prominent_min_coverage: float = 0.02
    wall_dominant_min_near_field_occupancy: float = 0.16
    wall_dominant_min_contrast: float = 0.35
    wall_dominant_max_target_contrast: float = 0.04
    sparse_empty_fraction: float = 0.82
    sparse_non_background_max: float = 0.18


@dataclass(frozen=True, slots=True)
class PhysicsTuningProfile:
    target_focus_sigma_mm: float = 4.5
    log_compression_gain_factor: float = 6.0
    normalization_reference_percentile: float = 99.5
    normalization_aux_percentile: float = 98.5
    normalization_spike_ratio: float = 1.22
    normalization_aux_blend_weight: float = 0.45
    sparse_signal_threshold: float = 0.01
    sparse_support_empty_fraction: float = 0.90
    wall_guardrail_empty_fraction: float = 0.86
    sparse_support_floor_base: float = 0.034
    sparse_support_floor_scale: float = 0.028
    sparse_support_anatomy_weight: float = 0.014
    sparse_support_target_weight: float = 0.018
    wall_guardrail_floor_base: float = 0.032
    wall_guardrail_floor_scale: float = 0.020
    wall_guardrail_anatomy_weight: float = 0.008
    wall_guardrail_target_weight: float = 0.010
    wall_guardrail_moderation: float = 0.94
    speckle_strength_default: float = 0.18
    reverberation_strength_default: float = 0.30
    shadow_strength_default: float = 0.42


@dataclass(frozen=True, slots=True)
class LoadedConsistencyProfile:
    name: str
    source_path: str
    settings: ConsistencyTuningProfile


@dataclass(frozen=True, slots=True)
class LoadedPhysicsProfile:
    name: str
    source_path: str
    settings: PhysicsTuningProfile


def _ensure_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}.")
    return value


def _resolve_profiles_path(path: str | Path | None = None) -> Path:
    resolved = DEFAULT_RENDER_PROFILES_PATH if path is None else Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Render profile config not found: {resolved}")
    return resolved


@lru_cache(maxsize=None)
def _load_profiles_payload_cached(path_str: str) -> Mapping[str, object]:
    path = Path(path_str)
    payload = yaml.safe_load(path.read_text())
    return _ensure_mapping(payload, label=f"render profile config {path}")


def _allowed_field_names(dataclass_type: type[object]) -> set[str]:
    return {field.name for field in fields(dataclass_type)}


def _build_profile_settings(dataclass_type: type[object], payload: Mapping[str, object], *, label: str):
    unknown_keys = sorted(set(payload) - _allowed_field_names(dataclass_type))
    if unknown_keys:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(unknown_keys)}")
    merged = asdict(dataclass_type())
    merged.update(payload)
    return dataclass_type(**merged)


def _validate_consistency_profile(profile: ConsistencyTuningProfile, *, label: str) -> None:
    _validate_fraction(profile.signal_threshold, label=f"{label}.signal_threshold")
    _validate_fraction(profile.near_field_fraction, label=f"{label}.near_field_fraction")
    _validate_positive(profile.target_region_radius_mm, label=f"{label}.target_region_radius_mm")
    _validate_fraction(profile.target_prominent_min_contrast, label=f"{label}.target_prominent_min_contrast")
    _validate_fraction(profile.target_prominent_min_coverage, label=f"{label}.target_prominent_min_coverage")
    _validate_fraction(profile.wall_dominant_min_near_field_occupancy, label=f"{label}.wall_dominant_min_near_field_occupancy")
    _validate_non_negative(profile.wall_dominant_min_contrast, label=f"{label}.wall_dominant_min_contrast")
    _validate_non_negative(profile.wall_dominant_max_target_contrast, label=f"{label}.wall_dominant_max_target_contrast")
    _validate_fraction(profile.sparse_empty_fraction, label=f"{label}.sparse_empty_fraction")
    _validate_fraction(profile.sparse_non_background_max, label=f"{label}.sparse_non_background_max")


def _validate_physics_profile(profile: PhysicsTuningProfile, *, label: str) -> None:
    _validate_positive(profile.target_focus_sigma_mm, label=f"{label}.target_focus_sigma_mm")
    _validate_positive(profile.log_compression_gain_factor, label=f"{label}.log_compression_gain_factor")
    _validate_percentile(profile.normalization_reference_percentile, label=f"{label}.normalization_reference_percentile")
    _validate_percentile(profile.normalization_aux_percentile, label=f"{label}.normalization_aux_percentile")
    _validate_positive(profile.normalization_spike_ratio, label=f"{label}.normalization_spike_ratio")
    _validate_fraction(profile.normalization_aux_blend_weight, label=f"{label}.normalization_aux_blend_weight")
    _validate_non_negative(profile.sparse_signal_threshold, label=f"{label}.sparse_signal_threshold")
    _validate_fraction(profile.sparse_support_empty_fraction, label=f"{label}.sparse_support_empty_fraction")
    _validate_fraction(profile.wall_guardrail_empty_fraction, label=f"{label}.wall_guardrail_empty_fraction")
    _validate_non_negative(profile.sparse_support_floor_base, label=f"{label}.sparse_support_floor_base")
    _validate_non_negative(profile.sparse_support_floor_scale, label=f"{label}.sparse_support_floor_scale")
    _validate_non_negative(profile.sparse_support_anatomy_weight, label=f"{label}.sparse_support_anatomy_weight")
    _validate_non_negative(profile.sparse_support_target_weight, label=f"{label}.sparse_support_target_weight")
    _validate_non_negative(profile.wall_guardrail_floor_base, label=f"{label}.wall_guardrail_floor_base")
    _validate_non_negative(profile.wall_guardrail_floor_scale, label=f"{label}.wall_guardrail_floor_scale")
    _validate_non_negative(profile.wall_guardrail_anatomy_weight, label=f"{label}.wall_guardrail_anatomy_weight")
    _validate_non_negative(profile.wall_guardrail_target_weight, label=f"{label}.wall_guardrail_target_weight")
    _validate_fraction(profile.wall_guardrail_moderation, label=f"{label}.wall_guardrail_moderation")
    _validate_non_negative(profile.speckle_strength_default, label=f"{label}.speckle_strength_default")
    _validate_non_negative(profile.reverberation_strength_default, label=f"{label}.reverberation_strength_default")
    _validate_non_negative(profile.shadow_strength_default, label=f"{label}.shadow_strength_default")


def _validate_non_negative(value: float, *, label: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{label} must be >= 0.0, got {value!r}.")


def _validate_positive(value: float, *, label: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be > 0.0, got {value!r}.")


def _validate_fraction(value: float, *, label: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must be in [0.0, 1.0], got {value!r}.")


def _validate_percentile(value: float, *, label: str) -> None:
    if not 0.0 < float(value) <= 100.0:
        raise ValueError(f"{label} must be in (0.0, 100.0], got {value!r}.")


def list_physics_profile_names(path: str | Path | None = None) -> list[str]:
    payload = _load_profiles_payload_cached(str(_resolve_profiles_path(path)))
    profiles = _ensure_mapping(payload.get("physics_profiles", {}), label="physics_profiles")
    return sorted(str(name) for name in profiles)


def load_consistency_profile(
    name: str = DEFAULT_CONSISTENCY_PROFILE_NAME,
    *,
    path: str | Path | None = None,
) -> LoadedConsistencyProfile:
    resolved_path = _resolve_profiles_path(path)
    payload = _load_profiles_payload_cached(str(resolved_path))
    profiles = _ensure_mapping(payload.get("consistency_profiles", {}), label="consistency_profiles")
    if name not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(f"Unknown consistency profile {name!r}. Available: {available}")
    settings = _build_profile_settings(
        ConsistencyTuningProfile,
        _ensure_mapping(profiles[name], label=f"consistency profile {name!r}"),
        label=f"consistency profile {name!r}",
    )
    _validate_consistency_profile(settings, label=f"consistency profile {name!r}")
    return LoadedConsistencyProfile(
        name=name,
        source_path=str(resolved_path),
        settings=settings,
    )


def load_physics_profile(
    name: str | None = None,
    *,
    path: str | Path | None = None,
) -> LoadedPhysicsProfile:
    resolved_name = DEFAULT_PHYSICS_PROFILE_NAME if name is None else str(name)
    resolved_path = _resolve_profiles_path(path)
    payload = _load_profiles_payload_cached(str(resolved_path))
    profiles = _ensure_mapping(payload.get("physics_profiles", {}), label="physics_profiles")
    if resolved_name not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(f"Unknown physics profile {resolved_name!r}. Available: {available}")
    settings = _build_profile_settings(
        PhysicsTuningProfile,
        _ensure_mapping(profiles[resolved_name], label=f"physics profile {resolved_name!r}"),
        label=f"physics profile {resolved_name!r}",
    )
    _validate_physics_profile(settings, label=f"physics profile {resolved_name!r}")
    return LoadedPhysicsProfile(
        name=resolved_name,
        source_path=str(resolved_path),
        settings=settings,
    )


def resolve_physics_artifact_settings(
    profile: PhysicsTuningProfile,
    *,
    speckle_strength: float | None = None,
    reverberation_strength: float | None = None,
    shadow_strength: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    explicit_overrides: dict[str, float] = {}
    effective = {
        "speckle_strength": float(profile.speckle_strength_default if speckle_strength is None else max(0.0, float(speckle_strength))),
        "reverberation_strength": float(profile.reverberation_strength_default if reverberation_strength is None else max(0.0, float(reverberation_strength))),
        "shadow_strength": float(profile.shadow_strength_default if shadow_strength is None else max(0.0, float(shadow_strength))),
    }
    if speckle_strength is not None:
        explicit_overrides["speckle_strength"] = effective["speckle_strength"]
    if reverberation_strength is not None:
        explicit_overrides["reverberation_strength"] = effective["reverberation_strength"]
    if shadow_strength is not None:
        explicit_overrides["shadow_strength"] = effective["shadow_strength"]
    return effective, explicit_overrides
