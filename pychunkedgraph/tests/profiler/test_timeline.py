"""Tests for pychunkedgraph.profiler.timeline: the request scope and its record."""

import json
import threading
import tracemalloc
from types import SimpleNamespace

import pytest
from flask import Flask, current_app

from pychunkedgraph.app.segmentation.v1 import routes as v1_routes
from pychunkedgraph.profiler import main, spans, timeline


def _stored(tmp_path):
    root = tmp_path / "ws" / "graphene_diagnostics" / "g1" / "timing"
    return sorted(root.rglob("*.json"))


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A small app: one edit view, one edit view that raises, and one other view."""
    monkeypatch.setattr(main, "PROFILER_ENABLED", True)
    monkeypatch.setattr(spans, "install", lambda: 0)
    graph = SimpleNamespace(
        graph_id="g1",
        meta=SimpleNamespace(
            custom_data={},
            data_source=SimpleNamespace(WATERSHED=f"file://{tmp_path}/ws"),
        ),
    )
    app = Flask(__name__)

    @app.route("/merge/<table_id>")
    def handle_merge(table_id):
        current_app.table_id = table_id
        current_app.operation_id = 7
        with main.get_profiler().profile("work"):
            return "ok"

    @app.route("/split/<table_id>")
    def handle_split(table_id):
        current_app.table_id = table_id
        main.active_scope().operation_id = 9
        raise RuntimeError("boom")

    @app.route("/roots")
    def handle_roots():
        return "ok"

    assert timeline.init_app(app, {"g1": graph}.get) is True
    return app


def test_an_edit_leaves_one_timeline_only_once_its_response_has_closed(app, tmp_path):
    response = app.test_client().get("/merge/g1")
    assert response.status_code == 200
    assert _stored(tmp_path) == []
    response.close()

    (path,) = _stored(tmp_path)
    record = json.loads(path.read_text())
    assert path.name == "7.json" and path.parent.name == record["started"][:10]
    assert record["endpoint"] == "handle_merge" and record["status"] == 200
    assert record["operation_id"] == 7 and record["graph_id"] == "g1"
    assert [block["path"] for block in record["blocks"]] == ["request.work", "request"]
    root = record["blocks"][-1]
    assert record["total_s"] == root["elapsed_s"] > 0
    assert root["rss_peak_bytes"] >= root["rss_start_bytes"] > 0
    assert record["tallies"] == [] and record["tallies_complete"] is True
    assert main.active_scope() is None


def test_every_edit_view_is_a_view_the_routes_define():
    for name in timeline.EDIT_VIEWS:
        assert callable(getattr(v1_routes, name, None)), name


def test_a_request_that_is_no_edit_opens_no_scope(app, tmp_path):
    response = app.test_client().get("/roots")
    response.close()
    assert response.status_code == 200 and _stored(tmp_path) == []


@pytest.mark.parametrize("propagate", [False, True])
def test_a_raising_edit_closes_its_scope_and_is_stored_under_its_operation(
    app, tmp_path, propagate
):
    app.config["PROPAGATE_EXCEPTIONS"] = propagate
    client = app.test_client()
    if propagate:
        with pytest.raises(RuntimeError):
            client.get("/split/g1")
    else:
        client.get("/split/g1").close()

    (path,) = _stored(tmp_path)
    record = json.loads(path.read_text())
    assert path.name == "9.json"
    assert record["status"] == 500 and record["operation_id"] == 9
    assert main.active_scope() is None


def test_an_edit_on_a_table_with_no_open_graph_is_dropped_quietly(app, tmp_path):
    response = app.test_client().get("/merge/other")
    response.close()
    assert response.status_code == 200 and _stored(tmp_path) == []


def test_a_failing_hook_never_fails_its_request(app, monkeypatch):
    monkeypatch.setattr(main, "ScopeProfiler", lambda: 1 / 0)
    response = app.test_client().get("/merge/g1")
    assert response.status_code == 200
    assert main.active_scope() is None


def test_with_the_switch_off_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(main, "PROFILER_ENABLED", False)
    monkeypatch.setattr(spans, "install", lambda: pytest.fail("installed"))
    app = Flask(__name__)
    assert timeline.init_app(app, {}.get) is False
    assert not app.before_request_funcs and not app.after_request_funcs
    assert not app.teardown_request_funcs


def test_a_span_table_that_fails_to_install_leaves_the_app_untimed(monkeypatch):
    def broken():
        raise KeyError("a renamed target")

    monkeypatch.setattr(main, "PROFILER_ENABLED", True)
    monkeypatch.setattr(spans, "install", broken)
    app = Flask(__name__)
    assert timeline.init_app(app, {}.get) is False
    assert not app.before_request_funcs and not app.after_request_funcs


def test_a_block_on_the_process_profiler_joins_the_open_scope_untraced():
    # A module that bound the process profiler at import holds this object for good.
    held = main.get_profiler()
    before = list(held.blocks)
    profiler = main.ScopeProfiler()
    scope = main.open_scope(profiler)
    try:
        with held.profile("point_cloud", with_memory=True, with_rss=True):
            assert not tracemalloc.is_tracing()
    finally:
        assert main.close_scope(scope) is True
    assert [block.path for block in profiler.blocks] == ["point_cloud"]
    assert held.blocks == before


def test_overlapping_scopes_report_their_tallies_incomplete():
    first = main.open_scope(main.ScopeProfiler())
    second = main.open_scope(main.ScopeProfiler())
    assert main.close_scope(second) is False
    assert main.close_scope(first) is False
    assert main.active_scope() is None


def test_a_scope_traces_no_allocation_and_only_its_root_starts_a_thread():
    profiler = main.ScopeProfiler()
    before = threading.active_count()
    with profiler.profile(timeline.ROOT_BLOCK, with_rss=True):
        assert threading.active_count() == before + 1
        with profiler.profile("stage", sampled_rss=True):
            assert threading.active_count() == before + 1
            assert not tracemalloc.is_tracing()
    assert threading.active_count() == before
    stage, root = profiler.blocks
    assert stage.rss_start_bytes > 0
    assert root.rss_peak_bytes >= root.rss_start_bytes > 0
