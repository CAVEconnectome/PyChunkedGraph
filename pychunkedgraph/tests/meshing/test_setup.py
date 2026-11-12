"""Tests for pychunkedgraph.pipeline.meshing.setup.setup_mesh_meta."""

from types import SimpleNamespace

import pytest

from pychunkedgraph.pipeline.meshing import setup
from pychunkedgraph.pipeline.meshing.meta import MeshConfig


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


def _migrated_cg(dynamic_mesh_dir="gs://old/graphene_meshes/dynamic"):
    """A table whose watershed lives apart from its meshes."""
    mesh = {
        "dir": "graphene_meshes",
        "initial_mesh_dir": "gs://old/graphene_meshes/initial",
        "dynamic_mesh_dir": dynamic_mesh_dir,
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


def _config(dynamic_mesh_dir=None):
    return MeshConfig(
        dir="graphene_meshes",
        mip=2,
        max_layer=6,
        max_error=40,
        minishard_bits={2: 3},
        dynamic_mesh_dir=dynamic_mesh_dir,
    )


def test_rerun_keeps_keys_the_config_cannot_set(written):
    mesh = setup.setup_mesh_meta(_migrated_cg(), _config())
    assert mesh["initial_mesh_dir"] == "gs://old/graphene_meshes/initial"
    assert mesh["verify"] is True


def test_rerun_writes_the_spec_beside_the_initial_meshes(written):
    setup.setup_mesh_meta(_migrated_cg(), _config())
    assert list(written) == ["gs://old/graphene_meshes/info"]


def test_rerun_keeps_a_pinned_mesh_path_over_a_relative_dynamic_dir(written):
    mesh = setup.setup_mesh_meta(_migrated_cg(), _config("dynamic"))
    assert mesh["dynamic_mesh_dir"] == "gs://old/graphene_meshes/dynamic"


def test_rerun_takes_an_absolute_dynamic_dir_the_config_names(written):
    mesh = setup.setup_mesh_meta(_migrated_cg(), _config("gs://new/dynamic"))
    assert mesh["dynamic_mesh_dir"] == "gs://new/dynamic"


def test_rerun_replaces_a_relative_dynamic_dir_with_the_configs(written):
    mesh = setup.setup_mesh_meta(_migrated_cg("dynamic"), _config())
    assert mesh["dynamic_mesh_dir"] == "dynamic_g"
