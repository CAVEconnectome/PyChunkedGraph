from datetime import datetime, timedelta, timezone

from ...ingest.create.parent_layer import add_parent_chunk
from ...meshing.setup import derive_initial_ts
from ..helpers import SV, label, create_chunk, fake_timestamp


def test_derive_initial_ts_uses_stamped_boundary(gen_graph):
    """derive_initial_ts returns the stamped ingest boundary + 1s (strict `<` threshold)."""
    graph = gen_graph(n_layers=4)
    ts = datetime(2026, 6, 1, 12, 0, 30, tzinfo=timezone.utc)
    graph.meta.custom_data["earliest_ts"] = ts.isoformat()
    assert derive_initial_ts(graph) == int(ts.timestamp()) + 1


def test_derive_initial_ts_after_root_build(gen_graph):
    """Root-layer build stamps earliest_ts; derive consumes it (fails if it stays unset)."""
    graph = gen_graph(n_layers=4)
    fake_ts = fake_timestamp()
    create_chunk(
        graph,
        vertices=[label(graph, SV())],
        edges=[],
        timestamp=fake_ts,
    )
    add_parent_chunk(graph, 3, [0, 0, 0], n_processes=1)
    add_parent_chunk(graph, 4, [0, 0, 0], n_processes=1)
    assert "earliest_ts" in graph.meta.custom_data
    boundary = datetime.fromisoformat(graph.meta.custom_data["earliest_ts"])
    assert derive_initial_ts(graph) == int(boundary.timestamp()) + 1
