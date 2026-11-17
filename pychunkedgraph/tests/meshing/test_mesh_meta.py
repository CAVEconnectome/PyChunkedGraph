"""Tests for pychunkedgraph.meshing.mesh_meta.MeshMeta."""

import posixpath
from types import SimpleNamespace

import numpy as np
import pytest

from pychunkedgraph.meshing.mesh_meta import MeshMeta


def _cg(watershed, mesh):
    meta = SimpleNamespace(
        custom_data={"mesh": mesh},
        data_source=SimpleNamespace(WATERSHED=watershed),
    )
    return SimpleNamespace(meta=meta)


def _created(*seconds):
    return np.array([np.datetime64(one, "s") for one in seconds], dtype="datetime64[s]")


class TestByDynamicDir:
    """New node is read from dir its creation time falls in; each dir grouped once."""

    SOURCE = "gs://src/graphene_meshes/dynamic"
    COPY = {
        "dir": "graphene_meshes",
        "inherited_dynamic": [{"path": SOURCE, "until": 100}],
    }

    def test_a_table_without_layers_reads_every_node_from_its_own_dir(self):
        mm = MeshMeta(_cg("gs://ws/seg", {"dir": "graphene_meshes"}))
        groups = mm.by_dynamic_dir([5, 6], _created(1, 2))
        assert list(groups) == [mm.dynamic_path]
        assert list(groups[mm.dynamic_path]) == [5, 6]

    def test_a_copy_reads_nodes_its_source_made_from_the_source(self):
        mm = MeshMeta(_cg("gs://ws/seg", self.COPY))
        groups = mm.by_dynamic_dir([5, 6, 7], _created(99, 100, 101))
        assert list(groups[self.SOURCE]) == [5]
        assert list(groups[mm.dynamic_path]) == [6, 7]

    def test_a_copy_of_a_copy_reads_each_node_from_the_table_that_made_it(self):
        older = "gs://src/graphene_meshes/dynamic_a"
        layers = [{"path": older, "until": 100}, {"path": self.SOURCE, "until": 200}]
        mesh = {"dir": "graphene_meshes", "inherited_dynamic": layers}
        mm = MeshMeta(_cg("gs://ws/seg", mesh))
        groups = mm.by_dynamic_dir([5, 6, 7], _created(99, 150, 200))
        assert list(groups[older]) == [5]
        assert list(groups[self.SOURCE]) == [6]
        assert list(groups[mm.dynamic_path]) == [7]

    def test_a_miss_searches_the_own_dir_then_inherited_dirs_newest_first(self):
        older = "gs://src/graphene_meshes/dynamic_a"
        layers = [{"path": older, "until": 100}, {"path": self.SOURCE, "until": 200}]
        mesh = {"dir": "graphene_meshes", "inherited_dynamic": layers}
        mm = MeshMeta(_cg("gs://ws/seg", mesh))
        assert mm.dynamic_stack == (mm.dynamic_path, self.SOURCE, older)

    def test_a_miss_searches_only_dirs_newer_than_its_hint(self):
        """An older dir's source never had the node, and may have given its id away."""
        oldest, middle = "gs://src/meshes/dynamic_a", "gs://src/meshes/dynamic_b"
        layers = [
            {"path": oldest, "until": 100},
            {"path": middle, "until": 200},
            {"path": self.SOURCE, "until": 300},
        ]
        mesh = {"dir": "graphene_meshes", "inherited_dynamic": layers}
        mm = MeshMeta(_cg("gs://ws/seg", mesh))
        own = mm.dynamic_path
        assert mm.fallback_dirs(oldest) == (own, self.SOURCE, middle)
        assert mm.fallback_dirs(self.SOURCE) == (own,)
        assert mm.fallback_dirs(own) == ()
        with pytest.raises(ValueError):
            mm.fallback_dirs("gs://elsewhere/dynamic")

    def test_no_nodes_group_to_nothing(self):
        """A copy given no nodes reads no creation times, as its caller passes none."""
        mm = MeshMeta(_cg("gs://ws/seg", self.COPY))
        assert mm.by_dynamic_dir([], None) == {}

    def test_a_copy_needs_v2(self):
        assert MeshMeta(_cg("gs://ws/seg", self.COPY)).needs_v2 is True


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


class TestSpecDir:
    """The shard spec is read beside the initial meshes, which a watershed elsewhere
    does not hold."""

    def test_watershed_elsewhere_keeps_spec_beside_initial(self):
        mesh = {
            "dir": "graphene_meshes",
            "initial_mesh_dir": "gs://old/graphene_meshes/initial",
        }
        mm = MeshMeta(_cg("gs://cheap/seg", mesh))
        assert mm.spec_dir == "gs://old/graphene_meshes"
