import numpy as np

from ebus_simulator.eval import summarize_bmode_regions


def test_summarize_bmode_regions_reports_region_contrast():
    image = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    sector = np.asarray(
        [
            [False, True, True],
            [True, True, True],
            [True, True, False],
        ],
        dtype=bool,
    )
    wall = np.asarray(
        [
            [False, False, True],
            [False, False, True],
            [False, False, False],
        ],
        dtype=bool,
    )
    vessel = np.asarray(
        [
            [False, False, False],
            [True, False, False],
            [True, False, False],
        ],
        dtype=bool,
    )
    target = np.asarray(
        [
            [False, False, False],
            [False, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )

    summary = summarize_bmode_regions(
        image,
        sector_mask=sector,
        target_mask=target,
        wall_mask=wall,
        vessel_mask=vessel,
    )

    assert summary["sector"]["pixel_count"] == 7
    assert np.isclose(summary["target"]["mean"], 0.65)
    assert np.isclose(summary["wall"]["mean"], 0.45)
    assert np.isclose(summary["vessel"]["mean"], 0.55)
    assert summary["target_contrast_vs_sector"] > 0.0
