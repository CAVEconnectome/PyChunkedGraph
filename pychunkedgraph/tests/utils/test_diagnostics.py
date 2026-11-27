"""Tests for pychunkedgraph.utils.diagnostics and the error dump that writes through it."""

from types import SimpleNamespace

import numpy as np
import pytest

from pychunkedgraph.graph import err_dump
from pychunkedgraph.utils import diagnostics


def _graph(watershed=None, mesh_dir=None):
    custom = {} if mesh_dir is None else {"mesh": {"dir": mesh_dir}}
    return SimpleNamespace(
        graph_id="g1",
        meta=SimpleNamespace(
            custom_data=custom, data_source=SimpleNamespace(WATERSHED=watershed)
        ),
    )


@pytest.mark.parametrize(
    "watershed, mesh_dir, parent",
    [
        ("gs://b/ws", "graphene_meshes", "gs://b/ws"),
        ("gs://b/ws/", None, "gs://b/ws"),
        ("gs://b/ws", "meshes/v2", "gs://b/ws/meshes"),
        ("gs://ro/ws", "gs://rw/m/graphene_meshes", "gs://rw/m"),
        (None, "gs://rw/m/graphene_meshes/", "gs://rw/m"),
    ],
)
def test_the_store_is_a_sibling_of_the_mesh_dir(watershed, mesh_dir, parent):
    cg = _graph(watershed, mesh_dir)
    assert (
        diagnostics.store_dir(cg, "errors")
        == f"{parent}/graphene_diagnostics/g1/errors"
    )


def test_a_bare_mesh_dir_with_no_watershed_has_no_store():
    cg = _graph(None, "graphene_meshes")
    assert diagnostics.store_dir(cg, "errors") is None
    assert diagnostics.put_json(cg, "errors", "1.json", {"a": 1}) is None
    assert diagnostics.get_json(cg, "errors", "1.json") is None


def test_an_object_written_is_read_back(tmp_path):
    cg = _graph(f"file://{tmp_path}/ws", "graphene_meshes")
    url = diagnostics.put_json(cg, "timing", "2026-01-01/7.json", {"total_s": 1.5})
    store = f"file://{tmp_path}/ws/graphene_diagnostics/g1/timing"
    assert url == f"{store}/2026-01-01/7.json"
    assert diagnostics.get_json(cg, "timing", "2026-01-01/7.json") == {"total_s": 1.5}
    assert diagnostics.get_json(cg, "timing", "2026-01-01/8.json") is None


def test_a_graph_whose_meta_cannot_be_read_fails_no_write_or_read():
    class Broken:
        graph_id = "g1"

        @property
        def meta(self):
            raise RuntimeError("no meta")

    assert diagnostics.put_json(Broken(), "errors", "1.json", {}) is None
    assert diagnostics.get_json(Broken(), "errors", "1.json") is None


def test_an_error_dump_lands_under_errors_and_reads_back(tmp_path):
    meshes = f"file://{tmp_path}/meshes/graphene_meshes"
    cg = _graph(f"file://{tmp_path}/ws", meshes)
    payload = {"op_id": 42, "ids": np.array([1, 2], dtype=np.uint64)}
    url = err_dump.dump_err_artifact(cg, 42, payload)
    assert url == f"file://{tmp_path}/meshes/graphene_diagnostics/g1/errors/42.json"
    assert err_dump.read_err_artifact(cg, 42) == {"op_id": 42, "ids": [1, 2]}
