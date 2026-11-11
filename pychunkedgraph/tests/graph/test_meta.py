"""Tests for pychunkedgraph.graph.meta.ChunkedGraphMeta.repoint_watershed."""

import pytest

from pychunkedgraph.graph import meta as meta_module
from pychunkedgraph.graph.meta import ChunkedGraphMeta, DataSource, GraphConfig
from pychunkedgraph.meshing.mesh_meta import MeshMeta

OLD, NEW = "gs://old/seg", "gs://new/seg"


@pytest.fixture
def infos(monkeypatch):
    """Watershed infos by path, served by a stub volume."""
    by_path = {OLD: {"at": OLD}, NEW: {"at": NEW}}

    class Volume:
        def __init__(self, path):
            self.info = by_path[path]

    monkeypatch.setattr(meta_module, "CloudVolume", Volume)
    return by_path


def _meta(custom_data):
    return ChunkedGraphMeta(GraphConfig(ID="g"), DataSource(WATERSHED=OLD), custom_data)


def test_repoint_reads_the_new_watershed(infos):
    meta = _meta({})
    assert meta.ws_info == {"at": OLD}
    assert meta.repoint_watershed(NEW) is meta
    assert meta.data_source.WATERSHED == NEW
    assert meta.ws_info == {"at": NEW}


def test_repoint_keeps_meshes_where_they_resolved(infos):
    mm = MeshMeta(_meta({"mesh": {"dir": "graphene_meshes"}}).repoint_watershed(NEW))
    assert mm.initial_path == f"{OLD}/graphene_meshes/initial"
    assert mm.dynamic_path == f"{OLD}/graphene_meshes/dynamic"


def test_repoint_keeps_the_old_watershed_grid_keys(infos):
    infos[OLD] = {
        "mesh_metadata": {"uniform_draco_grid_size": 7.0, "max_meshed_layer": 9}
    }
    infos[NEW] = {"mesh_metadata": {"max_meshed_layer": 3}}
    meta = _meta({"mesh": {"dir": "graphene_meshes", "max_meshed_layer": 5}})
    client = MeshMeta(meta.repoint_watershed(NEW)).client_metadata
    assert client["uniform_draco_grid_size"] == 7.0
    assert client["max_meshed_layer"] == 5


@pytest.mark.parametrize("custom_data", [{"READ_ONLY": True}, {"mesh": {"mip": 2}}])
def test_repoint_of_a_table_without_meshes_keeps_its_custom_data(infos, custom_data):
    meta = _meta(dict(custom_data)).repoint_watershed(NEW)
    assert meta.custom_data == custom_data
