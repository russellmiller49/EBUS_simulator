from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class PhysicsArtifactConfig:
    speckle_strength: float = 0.18
    reverberation_strength: float = 0.30
    shadow_strength: float = 0.42


@dataclass(slots=True)
class PhysicsArtifactMaps:
    speckle_map: np.ndarray
    reverberation_map: np.ndarray
    shadow_map: np.ndarray


def build_speckle_map(shape: tuple[int, int], *, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.ones(shape, dtype=np.float32)
    raw = rng.rayleigh(scale=1.0, size=shape).astype(np.float32)
    normalized = raw / max(float(raw.mean()), 1e-6)
    return np.clip(1.0 + (float(strength) * (normalized - 1.0)), 0.0, None).astype(np.float32)


def build_reverberation_map(
    interface_map: np.ndarray,
    *,
    depth_step_mm: float,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return np.zeros_like(interface_map, dtype=np.float32)

    interface = np.asarray(interface_map, dtype=np.float32)
    output = np.zeros_like(interface, dtype=np.float32)
    echo_offsets_mm = (1.4, 2.8, 4.2, 5.6)
    for index, offset_mm in enumerate(echo_offsets_mm, start=1):
        offset_px = max(1, int(round(offset_mm / max(float(depth_step_mm), 1e-6))))
        shifted = np.pad(interface[:-offset_px, :], ((offset_px, 0), (0, 0)), mode="constant") if offset_px < interface.shape[0] else np.zeros_like(interface)
        weight = float(strength) * (0.55 ** (index - 1))
        output += weight * shifted
    return ndimage.gaussian_filter(output, sigma=(0.9, 0.6)).astype(np.float32)


def build_shadow_map(
    *,
    airway_lumen_mask: np.ndarray,
    vessel_mask: np.ndarray,
    depth_step_mm: float,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return np.ones_like(airway_lumen_mask, dtype=np.float32)

    lumen = np.asarray(airway_lumen_mask, dtype=np.float32)
    vessel = np.asarray(vessel_mask, dtype=np.float32)
    blockers = (1.35 * lumen) + (0.30 * vessel)
    cumulative = np.cumsum(blockers, axis=0) * (float(depth_step_mm) / 10.0)
    return np.exp(-float(strength) * cumulative).astype(np.float32)


def apply_physics_artifacts(
    base_signal: np.ndarray,
    *,
    air_interface_map: np.ndarray,
    airway_lumen_mask: np.ndarray,
    vessel_mask: np.ndarray,
    depth_step_mm: float,
    config: PhysicsArtifactConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, PhysicsArtifactMaps]:
    signal = np.asarray(base_signal, dtype=np.float32)
    lumen = np.asarray(airway_lumen_mask, dtype=np.float32)
    speckle_map = build_speckle_map(signal.shape, rng=rng, strength=float(config.speckle_strength))
    reverberation_map = build_reverberation_map(
        air_interface_map,
        depth_step_mm=depth_step_mm,
        strength=float(config.reverberation_strength),
    )
    shadow_map = build_shadow_map(
        airway_lumen_mask=airway_lumen_mask,
        vessel_mask=vessel_mask,
        depth_step_mm=depth_step_mm,
        strength=float(config.shadow_strength),
    )
    intralumen_suppression = np.where(lumen > 0.0, 0.10, 1.0).astype(np.float32)
    reverberation_visible = reverberation_map * np.where(lumen > 0.0, 0.12, 1.0).astype(np.float32)
    output = np.clip((signal * speckle_map * shadow_map * intralumen_suppression) + reverberation_visible, 0.0, None).astype(np.float32)
    return output, PhysicsArtifactMaps(
        speckle_map=speckle_map,
        reverberation_map=reverberation_map,
        shadow_map=shadow_map,
    )
