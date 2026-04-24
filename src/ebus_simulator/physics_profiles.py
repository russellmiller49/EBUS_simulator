from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class PhysicsAppearanceProfile:
    name: str = "review_realistic_v1"
    tissue_floor: float = 0.028
    scatter_weight: float = 0.90
    boundary_weight: float = 1.70
    wall_boundary_boost: float = 0.26
    vessel_suppression: float = 0.38
    target_darkening_strength: float = 0.16
    target_rim_strength: float = 0.07
    tgc_start: float = 1.00
    tgc_end: float = 1.42
    log_gain: float = 7.5
    compression_percentile: float = 99.0
    output_gamma: float = 0.82
    sector_floor: float = 0.035
    post_blur_depth_sigma: float = 0.75
    post_blur_lateral_sigma: float = 0.38


DEFAULT_PHYSICS_APPEARANCE_PROFILE = PhysicsAppearanceProfile()


def _profile_from_mapping(payload: dict[str, object]) -> PhysicsAppearanceProfile:
    allowed = {field.name for field in fields(PhysicsAppearanceProfile)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown physics appearance profile field(s): {', '.join(unknown)}")
    values = asdict(DEFAULT_PHYSICS_APPEARANCE_PROFILE)
    values.update(payload)
    return PhysicsAppearanceProfile(**values)


def resolve_physics_appearance_profile(value: str | Path | None) -> PhysicsAppearanceProfile:
    if value is None or str(value).strip() in {"", "review_realistic", "review_realistic_v1", "default"}:
        return DEFAULT_PHYSICS_APPEARANCE_PROFILE

    candidate = Path(str(value)).expanduser()
    if candidate.exists():
        if candidate.suffix.lower() == ".json":
            payload = json.loads(candidate.read_text())
        else:
            payload = yaml.safe_load(candidate.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Physics appearance profile {candidate} must contain an object.")
        if "physics_appearance_profile" in payload:
            payload = payload["physics_appearance_profile"]
        if not isinstance(payload, dict):
            raise ValueError(f"Physics appearance profile {candidate} must contain a mapping.")
        return _profile_from_mapping(payload)

    raise ValueError(
        f"Unknown physics appearance profile {value!r}. "
        "Use 'review_realistic_v1' or pass a YAML/JSON profile path."
    )


def physics_profile_to_dict(profile: PhysicsAppearanceProfile) -> dict[str, object]:
    return asdict(profile)
