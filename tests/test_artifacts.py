import numpy as np

from ebus_simulator.artifacts import PhysicsArtifactConfig, apply_physics_artifacts, build_reverberation_map


def test_apply_physics_artifacts_is_seed_reproducible():
    base_signal = np.full((24, 12), 0.4, dtype=np.float32)
    air_interface = np.zeros((24, 12), dtype=np.float32)
    air_interface[4, 3:9] = 1.0
    lumen = np.zeros((24, 12), dtype=bool)
    lumen[5:12, 3:9] = True
    vessel = np.zeros((24, 12), dtype=bool)
    config = PhysicsArtifactConfig(speckle_strength=0.2, reverberation_strength=0.3, shadow_strength=0.4)

    first, first_maps = apply_physics_artifacts(
        base_signal,
        air_interface_map=air_interface,
        airway_lumen_mask=lumen,
        vessel_mask=vessel,
        depth_step_mm=0.8,
        config=config,
        rng=np.random.default_rng(7),
    )
    second, second_maps = apply_physics_artifacts(
        base_signal,
        air_interface_map=air_interface,
        airway_lumen_mask=lumen,
        vessel_mask=vessel,
        depth_step_mm=0.8,
        config=config,
        rng=np.random.default_rng(7),
    )
    third, _ = apply_physics_artifacts(
        base_signal,
        air_interface_map=air_interface,
        airway_lumen_mask=lumen,
        vessel_mask=vessel,
        depth_step_mm=0.8,
        config=config,
        rng=np.random.default_rng(8),
    )

    assert np.array_equal(first, second)
    assert np.array_equal(first_maps.speckle_map, second_maps.speckle_map)
    assert not np.array_equal(first, third)


def test_build_reverberation_map_adds_echoes_below_interface():
    interface = np.zeros((32, 8), dtype=np.float32)
    interface[3, 2:6] = 1.0

    reverberation = build_reverberation_map(interface, depth_step_mm=0.7, strength=0.4)

    assert float(reverberation[6:12, 2:6].mean()) > 0.0
    assert float(reverberation[0:2, 2:6].mean()) < 1e-4
