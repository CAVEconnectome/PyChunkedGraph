"""Tests for pychunkedgraph.meshing.manifest.dynamic, over real files in a copy's dirs."""

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from pychunkedgraph.meshing.manifest import dynamic
from pychunkedgraph.meshing.manifest.dynamic import (
    dynamic_buckets,
    hinted_rows,
    probe_exists,
    probe_read,
    search_dynamic,
)

#: Unix seconds of the copy's first open: a node made before it is the source's.
UNTIL = 100


class Dirs(NamedTuple):
    """A copy's watershed, its source's dynamic dir and its own, as cloud URLs."""

    watershed: str
    source: str
    own: str


def _cg(watershed, mesh, created=None):
    """A table stub; with ``created``, node id to Unix seconds, it reads timestamps."""
    meta = SimpleNamespace(
        custom_data={"mesh": mesh},
        data_source=SimpleNamespace(WATERSHED=watershed),
    )
    if created is None:
        return SimpleNamespace(meta=meta)

    def get_node_timestamps(node_ids):
        seconds = [created[int(i)] for i in node_ids]
        return np.array(seconds, dtype="datetime64[s]").astype("datetime64[us]")

    return SimpleNamespace(meta=meta, get_node_timestamps=get_node_timestamps)


def _copy(dirs, created):
    """A copy reading its source's dynamic dir for nodes made before ``UNTIL``."""
    layer = {"path": dirs.source, "until": UNTIL}
    mesh = {"dir": "graphene_meshes", "inherited_dynamic": [layer]}
    return _cg(dirs.watershed, mesh, created)


def _write(url, name, content=b"mesh"):
    """A mesh file ``name`` in the local dir ``url`` names."""
    (Path(url.removeprefix("file://")) / name).write_bytes(content)


@pytest.fixture(autouse=True)
def _names(monkeypatch):
    """File names from the node id alone; the chunk's box plays no part here."""
    monkeypatch.setattr(
        dynamic, "get_mesh_name", lambda cg, node_id: f"{int(node_id)}:0:x"
    )


@pytest.fixture
def dirs(tmp_path):
    source = tmp_path / "src" / "dynamic"
    own = tmp_path / "ws" / "graphene_meshes" / "dynamic"
    source.mkdir(parents=True)
    own.mkdir(parents=True)
    return Dirs(f"file://{tmp_path / 'ws'}", f"file://{source}", f"file://{own}")


class TestSearchDynamic:
    def test_a_node_its_source_made_is_found_in_the_sources_dir(self, dirs):
        _write(dirs.source, "5:0:x")
        found = search_dynamic(_copy(dirs, {5: 99}), [5], probe_exists)
        assert found[5].path == dirs.source

    def test_a_node_the_copy_made_is_found_in_its_own_dir(self, dirs):
        _write(dirs.own, "6:0:x")
        found = search_dynamic(_copy(dirs, {6: 101}), [6], probe_exists)
        assert found[6].path == dirs.own

    def test_a_mesh_the_copy_wrote_for_a_source_made_node_is_found_in_its_own_dir(
        self, dirs
    ):
        """The hint names the source, which lacks it, as after a remesh in the copy."""
        _write(dirs.own, "5:0:x")
        found = search_dynamic(_copy(dirs, {5: 99}), [5], probe_exists)
        assert found[5].path == dirs.own

    def test_a_node_in_no_dir_is_absent(self, dirs):
        assert search_dynamic(_copy(dirs, {7: 99}), [7], probe_exists) == {}

    def test_a_read_returns_the_mesh_bytes(self, dirs):
        _write(dirs.source, "5:0:x", b"draco")
        found = search_dynamic(_copy(dirs, {5: 99}), [5], probe_read)
        assert found[5].value == b"draco"

    def test_a_table_without_inherited_dirs_reads_no_timestamp(self, dirs):
        """The stub has no timestamp reader, so reading one would raise."""
        _write(dirs.own, "6:0:x")
        cg = _cg(dirs.watershed, {"dir": "graphene_meshes"})
        assert search_dynamic(cg, [6], probe_exists)[6].path == dirs.own


class TestRows:
    def test_a_speculative_row_names_the_sources_dir_for_a_node_it_made(self, dirs):
        rows = hinted_rows(_copy(dirs, {5: 99, 6: 101}), [5, 6])
        assert rows == [f"{dirs.source}/5:0:x", "6:0:x"]

    def test_a_row_lands_in_the_dir_it_names_and_a_bare_one_in_the_tables_own(
        self, dirs
    ):
        buckets = dynamic_buckets(_copy(dirs, {}), [f"{dirs.source}/5:0:x", "6:0:x"])
        assert buckets == {dirs.source: ["5:0:x"], dirs.own: ["6:0:x"]}

    def test_no_rows_have_no_buckets(self, dirs):
        assert dynamic_buckets(_copy(dirs, {}), []) == {}
