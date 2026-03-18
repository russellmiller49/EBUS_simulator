from ebus_simulator.device import _parse_branch_hint


def test_parse_branch_hint_supports_graph_scoped_tokens():
    hint = _parse_branch_hint("network:4, main:0, line:7")

    assert hint is not None
    assert hint.network_lines == frozenset({4})
    assert hint.main_lines == frozenset({0})
    assert hint.any_lines == frozenset({7})
