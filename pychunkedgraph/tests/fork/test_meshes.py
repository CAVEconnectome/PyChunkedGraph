"""A fork finds its source's meshes as of the fork point, and refuses to remesh."""

import json
from types import SimpleNamespace

import pytest
from flask import Flask

from ...app import app_utils
from ...app.meshing import common
from ...fork.guard import refuse_on_fork
from ...graph.exceptions import Forbidden
from ...meshing import meshgen
from ...meshing.manifest import sharded
from ...meshing.manifest.utils import (
    _get_dynamic_meshes,
    get_children_before_start_layer,
)
from ...meshing.mesh_meta import MeshMeta
from ...meshing.meshgen_utils import get_mesh_name


def _meshed(cg, dynamic_mesh_dir="dynamic"):
    """`cg` with a mesh block in its meta row; every node is newer than `initial_ts`."""
    block = {"dir": "mesh", "dynamic_mesh_dir": dynamic_mesh_dir, "initial_ts": 0}
    cg.meta._custom_data = {**cg.meta.custom_data, "mesh": block}
    cg.update_meta(cg.meta, overwrite=True)


def _manifest(cg, node_id, accept):
    """The meshing API's manifest for `node_id`, as `cg` answers it."""
    headers = {"Accept": accept} if accept else {}
    with Flask(__name__).test_request_context(headers=headers):
        return common.handle_get_manifest(cg.graph_id, str(node_id)).get_json()


def _never_started(*args, **kwargs):
    raise AssertionError("a refused remesh queued its thread")


def test_a_fork_finds_its_source_meshes_as_of_the_fork_point(
    split_pair, fork_of, tmp_path
):
    cg, ids, before = split_pair
    dynamic = tmp_path / "dynamic"
    dynamic.mkdir()
    _meshed(cg, dynamic.as_uri())
    fork = fork_of(cg, "test_fork", before)
    assert MeshMeta(fork).dynamic_path == MeshMeta(cg).dynamic_path == dynamic.as_uri()
    assert MeshMeta(fork).initial_path == MeshMeta(cg).initial_path
    root = cg.get_root(ids[0], time_stamp=before)
    assert fork.get_root(ids[0]) == root
    nodes = get_children_before_start_layer(fork, root, start_layer=2)
    for node in nodes:
        (dynamic / get_mesh_name(cg, node)).touch()
    found, missing = _get_dynamic_meshes(fork, nodes)
    assert set(found) == set(nodes) and missing.size == 0
    assert found == _get_dynamic_meshes(cg, nodes)[0]


@pytest.mark.parametrize("accept", [None, "application/x.cave;manifest_version=2"])
def test_a_fork_answers_its_source_manifest_as_of_the_fork_point(
    split_pair, fork_of, monkeypatch, accept
):
    cg, ids, before = split_pair
    _meshed(cg)
    fork = fork_of(cg, "test_fork", before)
    graphs = {cg.graph_id: cg, fork.graph_id: fork}
    monkeypatch.setattr(app_utils, "get_cg", lambda table_id, **_: graphs[table_id])
    # Shard readers place only nodes older than `initial_ts`, and none are; the emulator
    # graph has no segmentation info to build them from.
    no_shards = SimpleNamespace(mesh=SimpleNamespace(readers={}))
    monkeypatch.setattr(sharded, "CloudVolume", lambda *args, **kwargs: no_shards)
    root = cg.get_root(ids[0], time_stamp=before)
    assert fork.get_root(ids[0]) == root
    assert _manifest(fork, root, accept) == _manifest(cg, root, accept)


def test_remeshing_a_fork_is_refused_and_writes_nothing(split_pair, fork_of, tmp_path):
    cg, ids, before = split_pair
    fork = fork_of(cg, "test_fork", before)
    out = tmp_path / "out"
    with pytest.raises(Forbidden, match="read-only fork"):
        meshgen.remeshing(
            fork, fork.get_parents(ids), cv_unsharded_mesh_path=out.as_uri()
        )
    assert not out.exists()


def test_a_remesh_request_for_a_fork_answers_403_before_queueing(
    split_pair, fork_of, monkeypatch
):
    cg, ids, before = split_pair
    fork = fork_of(cg, "test_fork", before)
    monkeypatch.setattr(app_utils, "get_cg", lambda table_id, **_: fork)
    monkeypatch.setattr(common, "threading", SimpleNamespace(Thread=_never_started))
    body = json.dumps({"new_lvl2_ids": [int(i) for i in fork.get_parents(ids)]})
    with Flask(__name__).test_request_context(data=body):
        with pytest.raises(Forbidden) as refused:
            common.handle_remesh(fork.graph_id)
    assert refused.value.status_code == 403


def test_the_source_itself_is_not_refused(split_pair):
    cg, _, _ = split_pair
    refuse_on_fork(cg, "remeshing")
