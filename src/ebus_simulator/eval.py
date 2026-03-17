from __future__ import annotations

import numpy as np


def _region_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    region = np.asarray(values, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if region.size == 0:
        return {
            "pixel_count": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    return {
        "pixel_count": int(region.size),
        "mean": float(np.mean(region)),
        "std": float(np.std(region)),
        "p05": float(np.percentile(region, 5.0)),
        "p50": float(np.percentile(region, 50.0)),
        "p95": float(np.percentile(region, 95.0)),
    }


def summarize_bmode_regions(
    image: np.ndarray,
    *,
    sector_mask: np.ndarray,
    target_mask: np.ndarray | None = None,
    wall_mask: np.ndarray | None = None,
    vessel_mask: np.ndarray | None = None,
) -> dict[str, object]:
    signal = np.asarray(image, dtype=np.float32)
    sector = np.asarray(sector_mask, dtype=bool)
    summary: dict[str, object] = {
        "sector": _region_stats(signal, sector),
    }

    for name, mask in {
        "target": target_mask,
        "wall": wall_mask,
        "vessel": vessel_mask,
    }.items():
        if mask is None:
            summary[name] = _region_stats(signal, np.zeros_like(sector, dtype=bool))
            continue
        summary[name] = _region_stats(signal, np.asarray(mask, dtype=bool) & sector)

    sector_mean = summary["sector"]["mean"]
    for name in ("target", "wall", "vessel"):
        region_mean = summary[name]["mean"]
        summary[f"{name}_contrast_vs_sector"] = (
            None
            if region_mean is None or sector_mean is None
            else float(region_mean - sector_mean)
        )

    return summary
