from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AcousticField:
    impedance: np.ndarray
    scatter: np.ndarray
    attenuation: np.ndarray
    airway_lumen_mask: np.ndarray
    airway_wall_mask: np.ndarray
    vessel_mask: np.ndarray
    station_mask: np.ndarray
    target_focus: np.ndarray


def map_acoustic_properties(
    *,
    ct_hu: np.ndarray,
    airway_lumen_mask: np.ndarray,
    airway_wall_mask: np.ndarray,
    vessel_mask: np.ndarray,
    station_mask: np.ndarray | None = None,
    target_focus: np.ndarray | None = None,
) -> AcousticField:
    hu = np.asarray(ct_hu, dtype=np.float32)
    lumen = np.asarray(airway_lumen_mask, dtype=bool)
    wall = np.asarray(airway_wall_mask, dtype=bool)
    vessels = np.asarray(vessel_mask, dtype=bool)
    station = np.zeros_like(hu, dtype=bool) if station_mask is None else np.asarray(station_mask, dtype=bool)
    focus = np.zeros_like(hu, dtype=np.float32) if target_focus is None else np.clip(np.asarray(target_focus, dtype=np.float32), 0.0, 1.0)

    if lumen.shape != hu.shape or wall.shape != hu.shape or vessels.shape != hu.shape or station.shape != hu.shape or focus.shape != hu.shape:
        raise ValueError("Acoustic property inputs must all share the same shape.")

    tissue_fraction = np.clip((hu + 900.0) / 1400.0, 0.0, 1.0).astype(np.float32)

    impedance = (1.52 + (0.18 * tissue_fraction)).astype(np.float32)
    scatter = (0.16 + (0.28 * tissue_fraction)).astype(np.float32)
    attenuation = (0.42 + (0.18 * tissue_fraction)).astype(np.float32)

    if np.any(station):
        scatter = np.where(station, scatter * 0.72, scatter)
        attenuation = np.where(station, attenuation * 0.95, attenuation)

    if np.any(focus > 0.0):
        scatter *= (1.0 - (0.45 * focus))
        attenuation *= (1.0 - (0.12 * focus))
        impedance += (0.03 * focus)

    if np.any(vessels):
        impedance = np.where(vessels, 1.58, impedance)
        scatter = np.where(vessels, 0.035, scatter)
        attenuation = np.where(vessels, 0.30, attenuation)

    if np.any(wall):
        impedance = np.where(wall, 1.86 + (0.06 * tissue_fraction), impedance)
        scatter = np.where(wall, 0.82, scatter)
        attenuation = np.where(wall, 0.95, attenuation)

    if np.any(lumen):
        impedance = np.where(lumen, 0.04, impedance)
        scatter = np.where(lumen, 0.0, scatter)
        attenuation = np.where(lumen, 1.85, attenuation)

    return AcousticField(
        impedance=np.asarray(impedance, dtype=np.float32),
        scatter=np.asarray(scatter, dtype=np.float32),
        attenuation=np.asarray(attenuation, dtype=np.float32),
        airway_lumen_mask=lumen,
        airway_wall_mask=wall,
        vessel_mask=vessels,
        station_mask=station,
        target_focus=np.asarray(focus, dtype=np.float32),
    )
