from pathlib import Path

import ebus_simulator.localizer_renderer as localizer_renderer
import ebus_simulator.physics_renderer as physics_renderer
from ebus_simulator import rendering
from ebus_simulator.localizer_renderer import LOCALIZER_ENGINE_VERSION
from ebus_simulator.physics_renderer import PHYSICS_ENGINE_VERSION
from ebus_simulator.render_engines import RenderEngine, RenderRequest, RenderResult, parse_render_engine


def test_parse_render_engine_defaults_to_localizer():
    assert parse_render_engine(None) is RenderEngine.LOCALIZER
    assert parse_render_engine("localizer") is RenderEngine.LOCALIZER


def test_dispatch_render_request_routes_localizer(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_render_localizer_preset(request: RenderRequest, *, context=None) -> RenderResult:
        captured["engine"] = request.engine
        captured["output_path"] = request.output_path
        return RenderResult(
            engine=RenderEngine.LOCALIZER,
            engine_version=LOCALIZER_ENGINE_VERSION,
            rendered_preset="sentinel",
        )

    monkeypatch.setattr(localizer_renderer, "render_localizer_preset", fake_render_localizer_preset)
    result = rendering.dispatch_render_request(
        RenderRequest(
            manifest_path="case.yaml",
            preset_id="station_4r_node_b",
            output_path=tmp_path / "render.png",
        )
    )

    assert result.engine is RenderEngine.LOCALIZER
    assert result.engine_version == LOCALIZER_ENGINE_VERSION
    assert result.rendered_preset == "sentinel"
    assert captured["engine"] is RenderEngine.LOCALIZER
    assert captured["output_path"] == Path(tmp_path / "render.png")


def test_dispatch_render_request_routes_physics(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_render_physics_preset(request: RenderRequest, *, context=None) -> RenderResult:
        captured["engine"] = request.engine
        captured["output_path"] = request.output_path
        return RenderResult(
            engine=RenderEngine.PHYSICS,
            engine_version=PHYSICS_ENGINE_VERSION,
            rendered_preset="physics-sentinel",
        )

    monkeypatch.setattr(physics_renderer, "render_physics_preset", fake_render_physics_preset)
    result = rendering.dispatch_render_request(
        RenderRequest(
            manifest_path="case.yaml",
            preset_id="station_4r_node_b",
            output_path=tmp_path / "render.png",
            engine=RenderEngine.PHYSICS,
        )
    )

    assert result.engine is RenderEngine.PHYSICS
    assert result.engine_version == PHYSICS_ENGINE_VERSION
    assert result.rendered_preset == "physics-sentinel"
    assert captured["engine"] is RenderEngine.PHYSICS
    assert captured["output_path"] == Path(tmp_path / "render.png")
