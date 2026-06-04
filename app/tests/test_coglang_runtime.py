from app import coglang_runtime as cr


def test_extract_last_mexpr_picks_last_top_level():
    text = 'noise Traverse["a", "r"] more List["x"] trailing'
    assert cr.extract_last_mexpr(text) == 'List["x"]'


def test_extract_last_mexpr_skips_nested():
    # the LAST *top-level* expression, not the innermost nested head
    text = 'Trace[Update["x", {"a": 1}]]'
    assert cr.extract_last_mexpr(text) == 'Trace[Update["x", {"a": 1}]]'


def test_is_write_detects_nested_write():
    expr = cr.parse_expr('Trace[Update["warfarin", {"confidence": 0.5}]]')
    assert cr.is_write(expr) is True


def test_is_write_false_for_read():
    expr = cr.parse_expr('Traverse["warfarin", "interacts_with"]')
    assert cr.is_write(expr) is False


def test_format_result_empty_list():
    from coglang import parse, PythonCogLangExecutor
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("warfarin", category="Drug", name="Warfarin", confidence=1.0)
    ex = PythonCogLangExecutor(g)
    result = ex.execute(parse('Traverse["warfarin", "interacts_with"]'))
    out = cr.format_result(result)
    assert "empty" in out.lower()
