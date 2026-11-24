"""Manifest endpoint tests: pychunkedgraph.app.meshing.common.handle_get_manifest."""

from types import SimpleNamespace

import flask
import pytest

from pychunkedgraph.app import app_utils
from pychunkedgraph.app.meshing import common
from pychunkedgraph.meshing import manifest
from pychunkedgraph.meshing.manifest.v2 import MANIFEST_VERSION

#: A graph whose meshes lie under its watershed, so either version may serve it.
GRAPH = SimpleNamespace(
    meta=SimpleNamespace(
        custom_data={"mesh": {"dir": "graphene_meshes"}},
        data_source=SimpleNamespace(WATERSHED="gs://bucket/ws"),
    )
)


@pytest.fixture
def get_manifest(monkeypatch):
    """The endpoint's response to one ``Accept``, its graph load and shard reads
    stubbed: the I/O boundaries, so negotiation, assembly and headers run for real."""
    monkeypatch.setattr(app_utils, "get_cg", lambda table_id: GRAPH)
    monkeypatch.setattr(
        manifest, "speculative_manifest_sharded", lambda cg, node_id, **kw: ([], [])
    )

    def respond(accept: str):
        with flask.Flask(__name__).test_request_context(headers={"Accept": accept}):
            return common.handle_get_manifest("g", "1")

    return respond


def test_a_request_past_the_newest_is_told_the_newest(get_manifest):
    newest = MANIFEST_VERSION
    response = get_manifest(f"application/x.cave;manifest_version={newest + 1}")
    told = (response.headers["X-Manifest-Version"], response.json["manifest_version"])
    assert told == (str(newest), newest)
