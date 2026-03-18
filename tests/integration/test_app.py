from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ebus_simulator.app import PresetBrowserSession


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_preset_browser_session_renders_sector_and_context():
    session = PresetBrowserSession(MANIFEST_PATH, width=64, height=64)
    try:
        state = replace(
            session.default_state(),
            preset_id="station_4r_node_b",
            approach="default",
            engine="physics",
            overlay_vessels=True,
        )

        rendered = session.render(state)

        assert rendered.preset_id == "station_4r_node_b"
        assert rendered.approach == "default"
        assert rendered.engine == "physics"
        assert rendered.sector_rgb.shape == (64, 64, 3)
        assert rendered.context_rgb.shape == (64, 64, 3)
        assert Path(rendered.sector_metadata_path).exists()
        assert Path(rendered.context_metadata_path).exists()
        assert rendered.summary_text
        assert "Render Settings" in rendered.summary_text
        assert "- Station: Station 4R" in rendered.summary_text
        assert "- Normalization:" in rendered.summary_text
        assert rendered.inspector_sections
        assert rendered.screenshot_name_hint.endswith("_browser.png")
    finally:
        session.close()


def test_preset_browser_session_station_7_approaches_remain_distinct_in_inspector():
    session = PresetBrowserSession(MANIFEST_PATH, width=64, height=64)
    try:
        lms = session.render(
            replace(
                session.default_state(),
                preset_id="station_7_node_a",
                approach="lms",
                engine="physics",
            )
        )
        rms = session.render(
            replace(
                session.default_state(),
                preset_id="station_7_node_a",
                approach="rms",
                engine="physics",
            )
        )

        assert "- Station: Station 7" in lms.summary_text
        assert "- Approach: LMS" in lms.summary_text
        assert "- Station: Station 7" in rms.summary_text
        assert "- Approach: RMS" in rms.summary_text
        assert lms.summary_text != rms.summary_text
    finally:
        session.close()
