import numpy as np

from ebus_simulator.acoustic_properties import map_acoustic_properties


def test_map_acoustic_properties_applies_material_precedence_and_target_focus():
    ct_hu = np.asarray(
        [
            [40.0, 40.0, 40.0],
            [40.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    airway_lumen = np.asarray(
        [
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    airway_wall = np.asarray(
        [
            [True, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    vessel = np.asarray(
        [
            [False, False, True],
            [False, False, False],
        ],
        dtype=bool,
    )
    station = np.asarray(
        [
            [False, False, False],
            [False, True, False],
        ],
        dtype=bool,
    )
    target_focus = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    field = map_acoustic_properties(
        ct_hu=ct_hu,
        airway_lumen_mask=airway_lumen,
        airway_wall_mask=airway_wall,
        vessel_mask=vessel,
        station_mask=station,
        target_focus=target_focus,
    )

    assert field.scatter[0, 1] == 0.0
    assert field.impedance[0, 1] < 0.1
    assert field.scatter[0, 0] > field.scatter[0, 2]
    assert field.scatter[1, 1] < field.scatter[1, 2]
    assert field.attenuation[0, 1] > field.attenuation[1, 2]
