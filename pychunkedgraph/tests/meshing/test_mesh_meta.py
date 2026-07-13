"""Tests for pychunkedgraph.meshing.mesh_meta.MeshMeta."""

import posixpath
from types import SimpleNamespace

from pychunkedgraph.meshing.mesh_meta import MeshMeta


def _cg(watershed, mesh):
    meta = SimpleNamespace(
        custom_data={"mesh": mesh},
        data_source=SimpleNamespace(WATERSHED=watershed),
    )
    return SimpleNamespace(meta=meta)


class TestReaderAnchor:
    """reader_anchor must recompose to initial_path via cloud-volume's hardcoded
    ``initial`` subdir, so the verified shard reader reads the real bucket."""

    def test_colocated_anchor_is_watershed_dir(self):
        mm = MeshMeta(_cg("gs://ws/seg", {"dir": "graphene_meshes"}))
        assert mm.reader_anchor == ("gs://ws/seg", "graphene_meshes")
        assert posixpath.join(*mm.reader_anchor, "initial") == mm.initial_path
        assert mm.needs_v2 is False

    def test_migrated_initial_anchor_recomposes(self):
        mesh = {
            "dir": "graphene_meshes",
            "initial_mesh_dir": "gs://old/graphene_meshes/initial",
        }
        mm = MeshMeta(_cg("gs://cheap/seg", mesh))
        assert mm.initial_path == "gs://old/graphene_meshes/initial"
        assert posixpath.join(*mm.reader_anchor, "initial") == mm.initial_path
        assert mm.needs_v2 is True
