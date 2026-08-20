"""Tests for the optional pcgl2cache acceleration of ``find_path``.

``compute_coordinate_path`` replaces what ``pathing.compute_rough_coordinate_path``
returns -- the geometric center of each level 2 node's chunk -- with pcgl2cache's
``rep_coord_nm`` wherever the cache has one. The point of these tests is the fallback
contract rather than the happy path: the endpoint must return the same number of points,
in the same units, and must never surface an l2cache failure to the caller. Every path
below therefore asserts that the chunk centers survive when something goes wrong.

Mocking happens at the ``L2CacheClient`` boundary, and ``pathing`` is stubbed out so the
suite does not need graph_tool (conda-only) to run.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from flask import Flask, g


def _load_l2cache_utils():
    """Load pychunkedgraph.app.l2cache_utils without executing pychunkedgraph.app.

    ``pychunkedgraph/app/__init__.py`` registers every blueprint, which drags in the whole
    server stack (cloud-volume and friends) -- not installable in the dev venv, and none of
    it needed here. Loading the file directly avoids that. The name it is registered under
    keeps ``__package__`` correct, which is what the lazy ``from ..graph.analysis import
    pathing`` inside ``compute_coordinate_path`` resolves against.
    """
    name = "pychunkedgraph.app.l2cache_utils"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "app" / "l2cache_utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


l2cache_utils = _load_l2cache_utils()

ROUGH_PATH = [
    np.array([100.0, 100.0, 100.0]),
    np.array([200.0, 200.0, 200.0]),
    np.array([300.0, 300.0, 300.0]),
]
L2_PATH = np.array([11, 22, 33], dtype=np.uint64)
TABLE_ID = "test_table"
SERVER = "http://pcgl2cache-service"


class FakeCg:
    graph_id = TABLE_ID


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["L2CACHE_URL"] = SERVER
    app.config["AUTH_TOKEN"] = "service-token"
    return app


@pytest.fixture(autouse=True)
def clear_mapping_cache():
    l2cache_utils._MAPPING_CACHE.clear()
    yield
    l2cache_utils._MAPPING_CACHE.clear()


@pytest.fixture(autouse=True)
def stub_pathing(monkeypatch):
    """Stand in for pychunkedgraph.graph.analysis.pathing, which imports graph_tool.

    ``compute_coordinate_path`` imports it lazily, so seeding sys.modules with the parent
    package is enough to satisfy the import without touching disk. A fresh copy of
    ROUGH_PATH is returned each call so in-place overwrites cannot leak between tests.
    """
    pathing = types.ModuleType("pychunkedgraph.graph.analysis.pathing")
    pathing.compute_rough_coordinate_path = lambda cg, l2_ids: [c.copy() for c in ROUGH_PATH]
    analysis = types.ModuleType("pychunkedgraph.graph.analysis")
    analysis.pathing = pathing
    monkeypatch.setitem(sys.modules, "pychunkedgraph.graph.analysis", analysis)
    return pathing


class FakeClient:
    """Minimal stand-in for caveclient's L2CacheClient."""

    def __init__(self, mapping=None, data=None, mapping_exc=None, data_exc=None):
        self._mapping = mapping if mapping is not None else {TABLE_ID: {}}
        self._data = data if data is not None else {}
        self._mapping_exc = mapping_exc
        self._data_exc = data_exc
        self.mapping_calls = 0
        self.data_calls = 0

    def table_mapping(self):
        self.mapping_calls += 1
        if self._mapping_exc is not None:
            raise self._mapping_exc
        return self._mapping

    def get_l2data(self, l2_ids, attributes=None):
        self.data_calls += 1
        if self._data_exc is not None:
            raise self._data_exc
        return self._data


@pytest.fixture
def use_client(monkeypatch):
    """Install a FakeClient as the return of _client and hand it back for assertions."""

    def _install(client):
        calls = []

        def fake_client(table_id):
            calls.append(table_id)
            return client

        monkeypatch.setattr(l2cache_utils, "_client", fake_client)
        return calls

    return _install


def test_feature_disabled_never_builds_a_client(app, monkeypatch):
    """L2CACHE_URL unset is the off switch: identical output, no client, no request."""
    app.config["L2CACHE_URL"] = None

    def explode(table_id):
        raise AssertionError("_client must not be called when L2CACHE_URL is unset")

    monkeypatch.setattr(l2cache_utils, "_client", explode)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert [c.tolist() for c in result] == [c.tolist() for c in ROUGH_PATH]


def test_caveclient_unavailable_falls_back(app, monkeypatch):
    monkeypatch.setattr(l2cache_utils, "_client", lambda table_id: None)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert [c.tolist() for c in result] == [c.tolist() for c in ROUGH_PATH]


def test_table_without_a_cache_skips_the_attributes_request(app, use_client):
    client = FakeClient(mapping={"some_other_table": {}})
    use_client(client)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert [c.tolist() for c in result] == [c.tolist() for c in ROUGH_PATH]
    assert client.data_calls == 0


def test_all_ids_cached_replaces_every_point(app, use_client):
    client = FakeClient(
        data={
            "11": {"rep_coord_nm": [1, 2, 3]},
            "22": {"rep_coord_nm": [4, 5, 6]},
            "33": {"rep_coord_nm": [7, 8, 9]},
        }
    )
    use_client(client)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert len(result) == len(L2_PATH)
    assert [c.tolist() for c in result] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_partial_cache_keeps_chunk_centers_for_the_rest(app, use_client):
    """pcgl2cache returns an empty dict for an l2 id it has not computed yet."""
    client = FakeClient(
        data={
            "11": {"rep_coord_nm": [1, 2, 3]},
            "22": {},
            "33": {"rep_coord_nm": [7, 8, 9]},
        }
    )
    use_client(client)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert len(result) == len(L2_PATH)
    assert result[0].tolist() == [1, 2, 3]
    assert result[1].tolist() == ROUGH_PATH[1].tolist()
    assert result[2].tolist() == [7, 8, 9]


def test_computed_but_empty_node_falls_back(app, use_client):
    """A node written with the size_nm3=0 sentinel has no rep_coord_nm."""
    client = FakeClient(data={"11": {"size_nm3": 0}, "22": {}, "33": {}})
    use_client(client)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert [c.tolist() for c in result] == [c.tolist() for c in ROUGH_PATH]


def test_table_mapping_failure_falls_back(app, use_client):
    client = FakeClient(mapping_exc=RuntimeError("connection timed out"))
    use_client(client)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert [c.tolist() for c in result] == [c.tolist() for c in ROUGH_PATH]
    assert client.data_calls == 0


def test_table_mapping_failure_is_cached(app, use_client):
    """A down service must not cost every request a full round of timeouts."""
    client = FakeClient(mapping_exc=RuntimeError("connection timed out"))
    use_client(client)

    with app.test_request_context():
        for _ in range(3):
            l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert client.mapping_calls == 1


def test_attributes_failure_falls_back(app, use_client):
    client = FakeClient(data_exc=RuntimeError("503 from l2cache"))
    use_client(client)

    with app.test_request_context():
        result = l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert [c.tolist() for c in result] == [c.tolist() for c in ROUGH_PATH]


def test_attributes_failure_reprobes_the_mapping(app, use_client):
    """The mapping was fetched while the service was up; a data failure invalidates it."""
    client = FakeClient(data_exc=RuntimeError("503 from l2cache"))
    use_client(client)

    with app.test_request_context():
        l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)
        l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert client.mapping_calls == 2


def test_table_mapping_is_memoized_across_calls(app, use_client):
    client = FakeClient(data={"11": {"rep_coord_nm": [1, 2, 3]}})
    use_client(client)

    with app.test_request_context():
        l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)
        l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert client.mapping_calls == 1
    assert client.data_calls == 2


def test_expired_table_mapping_is_refetched(app, use_client):
    client = FakeClient(data={"11": {"rep_coord_nm": [1, 2, 3]}})
    use_client(client)
    app.config["L2CACHE_MAPPING_TTL_S"] = -1

    with app.test_request_context():
        l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)
        l2cache_utils.compute_coordinate_path(FakeCg(), L2_PATH)

    assert client.mapping_calls == 2


def test_caller_token_is_preferred_over_the_service_token(app):
    with app.test_request_context():
        g.auth_token = "caller-token"
        assert l2cache_utils._auth_token() == "caller-token"


@pytest.mark.parametrize("token", [None, "AUTH_DISABLED"])
def test_service_token_used_when_there_is_no_caller_token(app, token):
    with app.test_request_context():
        g.auth_token = token
        assert l2cache_utils._auth_token() == "service-token"
