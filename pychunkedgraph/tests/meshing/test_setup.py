from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ...ingest.create.parent_layer import add_parent_chunk
from ...meshing import setup
from ...meshing.meta import MeshConfig
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


@pytest.fixture
def written(monkeypatch):
    """Spec writes by path, with draco and block-shape lookups stubbed."""
    specs = {}

    class Files:
        def __init__(self, root):
            self.root = root

        def put_json(self, name, data, cache_control=None):
            specs[f"{self.root}/{name}"] = data

    monkeypatch.setattr(setup, "CloudFiles", Files)
    monkeypatch.setattr(
        setup, "get_mesh_block_shape_for_mip", lambda *_: (256, 256, 128)
    )
    monkeypatch.setattr(
        setup,
        "get_draco_encoding_settings_for_chunk",
        lambda *_, **__: {"quantization_range": 1023.0, "quantization_bits": 10},
    )
    return specs


def _migrated_cg():
    """A table whose watershed lives apart from its initial meshes."""
    mesh = {
        "dir": "graphene_meshes",
        "initial_mesh_dir": "gs://old/graphene_meshes/initial",
        "initial_ts": 1700000000,
        "verify": True,
    }
    meta = SimpleNamespace(
        custom_data={"mesh": mesh},
        data_source=SimpleNamespace(WATERSHED="gs://cheap/seg"),
        ws_cv=SimpleNamespace(info={"scales": [{}, {}, {}]}),
    )
    return SimpleNamespace(
        graph_id="g",
        meta=meta,
        get_chunk_id=lambda **_: 0,
        update_meta=lambda *_, **__: None,
    )


def _config():
    return MeshConfig(
        dir="graphene_meshes", mip=2, max_layer=6, max_error=40, minishard_bits={2: 3}
    )


def test_rerun_keeps_keys_the_config_cannot_set(written):
    mesh = setup.setup_mesh_meta(_migrated_cg(), _config())
    assert mesh["initial_mesh_dir"] == "gs://old/graphene_meshes/initial"
    assert mesh["verify"] is True


def test_rerun_writes_the_spec_beside_the_initial_meshes(written):
    setup.setup_mesh_meta(_migrated_cg(), _config())
    assert list(written) == ["gs://old/graphene_meshes/info"]
