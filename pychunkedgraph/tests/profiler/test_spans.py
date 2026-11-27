"""Tests for pychunkedgraph.profiler.spans: the span table and its installer."""

import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from pychunkedgraph.graph import cutting, operation
from pychunkedgraph.graph.chunkedgraph import ChunkedGraph
from pychunkedgraph.graph.sv_split import edits as sv_edits
from pychunkedgraph.graph.sv_split import splitter
from pychunkedgraph.graph.sv_split.state import SplitResult
from pychunkedgraph.profiler import main, spans

from ..helpers import SV, build_graph


def _span(name, memory=False, then=None):
    return spans.Span(name, "", memory, then)


@pytest.fixture
def installed():
    spans.install()
    yield
    spans.uninstall()


@pytest.fixture
def scope():
    profiler = main.ScopeProfiler()
    opened = main.open_scope(profiler)
    yield profiler
    main.close_scope(opened)


def test_every_span_names_a_function_its_owner_defines():
    for span in spans.SPANS:
        owner, attr = spans.resolve(span.target)
        assert attr in vars(owner), span.target
        original = vars(owner)[attr]
        assert callable(getattr(original, "__func__", original)), span.target


def test_no_span_reaches_inside_the_splitter_package():
    package = splitter.DEFAULT_SPLITTER.partition(".")[0]
    assert not [span.target for span in spans.SPANS if span.target.startswith(package)]


def test_install_wraps_each_target_once_and_uninstall_restores_it():
    before = vars(ChunkedGraph)["add_edges"]
    multicut = cutting.run_multicut
    try:
        assert spans.install() >= len(spans.SPANS)
        assert vars(ChunkedGraph)["add_edges"].pcg_span == "add_edges"
        # `operation` imported `run_multicut` by name, so its reference is wrapped too.
        assert operation.run_multicut is cutting.run_multicut
        assert operation.run_multicut.pcg_span == "multicut"
        assert spans.install() == 0
    finally:
        spans.uninstall()
    assert vars(ChunkedGraph)["add_edges"] is before
    assert operation.run_multicut is multicut and cutting.run_multicut is multicut


def test_a_wrapped_call_outside_a_scope_goes_straight_through():
    calls = []
    timed = spans.timed(
        _span("x", True), lambda value: calls.append(value) or value * 2
    )
    assert timed(3) == 6 and calls == [3]
    assert main.active_scope() is None


def test_nested_wrapped_calls_nest_their_paths_and_only_coarse_ones_read_rss(scope):
    inner = spans.timed(_span("inner"), lambda: 1)
    outer = spans.timed(_span("outer", True), lambda: inner() + inner())
    assert outer() == 2
    blocks = {block.path: block for block in scope.blocks}
    assert blocks["outer.inner"].call_count == 2
    assert blocks["outer"].rss_start_bytes > 0
    assert blocks["outer.inner"].rss_start_bytes == 0


def test_a_span_starts_no_thread_and_traces_no_allocation(scope):
    timed = spans.timed(_span("x", True), threading.active_count)
    before = threading.active_count()
    assert timed() == before
    assert not tracemalloc.is_tracing()


def test_a_pool_threads_call_is_tallied_and_leaves_the_owners_path_alone(scope):
    work = spans.timed(_span("work"), threading.get_ident)
    with scope.profile("stage"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            idents = list(pool.map(lambda _: work(), range(4)))
        assert scope.current_path == ["stage"]
    assert threading.get_ident() not in idents
    assert [(tally.name, tally.calls) for tally in scope.tallies] == [("work", 4)]
    assert [block.path for block in scope.blocks] == ["stage"]


def test_a_span_that_cannot_open_drops_itself_and_never_its_call(scope, monkeypatch):
    def unreadable():
        raise OSError("no statm")

    monkeypatch.setattr(scope, "sample_rss", unreadable)
    coarse = spans.timed(_span("coarse", True), lambda: "done")
    plain = spans.timed(_span("plain"), lambda: "next")
    assert coarse() == "done"
    # The failed span left the scope's open path as it was.
    assert scope.current_path == [] and plain() == "next"
    assert [block.path for block in scope.blocks] == ["plain"]


def test_a_span_that_cannot_close_still_returns_its_calls_result(scope, monkeypatch):
    reads = []

    def readable_once():
        reads.append(1)
        if len(reads) > 1:
            raise OSError("no statm")
        return 1024

    monkeypatch.setattr(scope, "sample_rss", readable_once)
    coarse = spans.timed(_span("coarse", True), lambda: "written")
    assert coarse() == "written"
    assert scope.current_path == []


def test_then_gets_the_result_once_the_span_has_closed(scope):
    seen = []
    timed = spans.timed(_span("x", then=seen.append), lambda: "graph")
    assert timed() == "graph" and seen == ["graph"]


def test_a_follow_up_that_raises_never_fails_its_call(scope):
    timed = spans.timed(_span("x", then=lambda result: 1 / 0), lambda: 5)
    assert timed() == 5


def test_the_get_cg_span_times_the_client_of_the_graph_it_returns(scope):
    class Client:
        def read_nodes(self):
            return 1

    graph = SimpleNamespace(client=Client())
    row = next(span for span in spans.SPANS if span.name == "get_cg")
    get_cg = spans.timed(row, lambda: graph)
    try:
        assert get_cg() is graph
        assert graph.client.read_nodes() == 1
        assert [block.path for block in scope.blocks] == ["get_cg", "bt.read_nodes"]
    finally:
        spans.uninstall()


def test_a_client_class_is_timed_once_and_restored_exactly(scope):
    class Client:
        def read_nodes(self):
            return 1

    class Inheriting(Client):
        pass

    original = vars(Client)["read_nodes"]
    try:
        spans.wrap_client(Inheriting)
        spans.wrap_client(Client)
        spans.wrap_client(Client)
        assert Client().read_nodes() == 1 and Inheriting().read_nodes() == 1
        assert [block.path for block in scope.blocks] == ["bt.read_nodes"]
        assert scope.blocks[0].call_count == 2
    finally:
        spans.uninstall()
    assert vars(Client)["read_nodes"] is original
    assert "read_nodes" not in vars(Inheriting)


@pytest.mark.timeout(60)
def test_a_merge_in_a_scope_times_its_stages_and_its_storage_calls(
    gen_graph, installed, scope
):
    cg, sv = build_graph(
        gen_graph,
        n_layers=2,
        atomic_chunk_bounds=np.array([1, 1, 1]),
        supervoxels={"a0": SV(), "a1": SV(seg=1)},
    )
    for name in spans.CLIENT_CALLS:
        assert hasattr(type(cg.client), name), name
    spans.wrap_client(type(cg.client))

    cg.add_edges("Jane Doe", [sv["a1"], sv["a0"]], affinities=[0.3])

    assert scope.operation_id is not None
    paths = [block.path for block in scope.blocks]
    for stage in ("lock", "apply", "write"):
        assert f"add_edges.execute.{stage}" in paths, paths
    assert any(path.endswith("rebuild") for path in paths), paths
    timed = paths + [tally.name for tally in scope.tallies]
    assert any(name.endswith("bt.read_nodes") for name in timed), timed
    assert any(name.endswith("bt.write") for name in timed), timed


@pytest.mark.timeout(60)
def test_a_split_that_needs_a_supervoxel_split_times_that_phase_under_apply(
    gen_graph, scope, monkeypatch
):
    cg, sv = build_graph(
        gen_graph,
        n_layers=3,
        supervoxels={"a0": SV(), "b": SV(x=1)},
        edges=[("a0", "b", 0.5)],
    )
    size = cg.meta.graph_config.CHUNK_SIZE
    edge = np.array([[sv["a0"], sv["b"]]], dtype=np.uint64)
    answers = iter([cutting.SvSplitRequired({}), cutting.Cut(edge)])

    def unsplit(_cg, *, source_ids, sink_ids, **_):
        return SplitResult([], source_ids, sink_ids, [], [])

    # The multicut asks for a supervoxel split once, and that split changes nothing.
    monkeypatch.setattr(operation, "run_multicut", lambda *_, **__: next(answers))
    monkeypatch.setattr(sv_edits, "plan_sv_splits", lambda *_, **__: ([], []))
    monkeypatch.setattr(sv_edits, "split_supervoxels", unsplit)
    spans.install()
    try:
        cg.remove_edges(
            "Jane Doe",
            source_ids=sv["a0"],
            sink_ids=sv["b"],
            source_coords=[0, 0, 0],
            sink_coords=[2 * size[0], 2 * size[1], size[2]],
            mincut=True,
        )
    finally:
        spans.uninstall()

    paths = [block.path for block in scope.blocks]
    for part in (
        "sv_split_plan",
        "l2_lock",
        "sv_split",
        "l2_lock_indefinite",
        "sv_split_write",
        "l2_unlock_indefinite",
        "l2_unlock",
    ):
        assert f"remove_edges.execute.apply.{part}" in paths, paths
