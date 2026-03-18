from __future__ import annotations

from ebus_simulator.render_profiles import (
    list_physics_profile_names,
    load_consistency_profile,
    load_physics_profile,
    resolve_physics_artifact_settings,
)


def test_render_profiles_load_expected_baseline_and_named_profile():
    consistency_profile = load_consistency_profile()
    baseline = load_physics_profile()
    sparse_support_boost = load_physics_profile("sparse_support_boost")

    assert consistency_profile.name == "baseline"
    assert consistency_profile.settings.target_region_radius_mm == 4.0
    assert baseline.name == "baseline"
    assert baseline.settings.log_compression_gain_factor == 6.0
    assert sparse_support_boost.name == "sparse_support_boost"
    assert sparse_support_boost.settings.sparse_support_floor_base > baseline.settings.sparse_support_floor_base
    assert sparse_support_boost.settings.speckle_strength_default < baseline.settings.speckle_strength_default
    assert list_physics_profile_names() == ["baseline", "sparse_support_boost"]


def test_resolve_physics_artifact_settings_applies_explicit_overrides():
    profile = load_physics_profile("sparse_support_boost").settings

    effective, explicit = resolve_physics_artifact_settings(
        profile,
        speckle_strength=0.22,
        reverberation_strength=None,
        shadow_strength=None,
    )

    assert effective == {
        "speckle_strength": 0.22,
        "reverberation_strength": 0.24,
        "shadow_strength": 0.36,
    }
    assert explicit == {"speckle_strength": 0.22}
