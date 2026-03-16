from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ebus_simulator.models import VolumeData

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def load_nifti(path: str | Path, *, kind: str, load_data: bool = False) -> VolumeData:
    resolved_path = Path(path).expanduser().resolve()
    image = nib.load(str(resolved_path))
    affine_ras = np.asarray(image.affine, dtype=np.float64)
    affine_lps = RAS_TO_LPS @ affine_ras
    data = np.asarray(image.dataobj) if load_data else None
    return VolumeData(
        path=resolved_path,
        kind=kind,
        shape=tuple(int(dim) for dim in image.shape),
        dtype=str(image.get_data_dtype()),
        affine_ras=affine_ras,
        affine_lps=affine_lps,
        inverse_affine_lps=np.linalg.inv(affine_lps),
        voxel_sizes_mm=np.asarray(image.header.get_zooms()[:3], dtype=np.float64),
        axis_codes_ras=tuple(str(code) for code in nib.aff2axcodes(affine_ras)),
        data=data,
    )
