"""Tests for app/readiness.py: the per-graph readiness report and the edit gate.

Why this exists: a graph's mesh output lives under its WATERSHED bucket, and the pipeline's worker
SA can write anywhere, so a missing IAM binding for the serving SA only surfaced when remeshing
silently failed after a user edit. readiness probes those writes as the serving identity and
blocks edits until they work.

readiness.py is loaded by path with its collaborators faked, so these tests run without
cloudfiles, cloudvolume, redis or a Flask app.
"""

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCloudFiles:
    """In-memory bucket store. Paths listed in DENY_WRITE refuse puts."""

    FILES = {}
    DENY_WRITE = set()

    def __init__(self, path):
        self.path = path.rstrip("/")

    def put(self, name, content, **kwargs):
        if any(self.path.startswith(p) for p in self.DENY_WRITE):
            raise PermissionError("403 storage.objects.create denied")
        self.FILES[f"{self.path}/{name}"] = content

    def delete(self, name):
        self.FILES.pop(f"{self.path}/{name}", None)

    def exists(self, name):
        return f"{self.path}/{name}" in self.FILES

    def list(self):
        return (k for k in self.FILES if k.startswith(self.path + "/"))


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.down = False

    def get(self, key):
        if self.down:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.down:
            raise ConnectionError("redis down")
        self.store[key] = value


def _load_readiness():
    """Load app/readiness.py, using the real graph package when importable.

    Without cloudvolume, `pychunkedgraph.graph` cannot be imported, so a stand-in exposing only the
    exceptions module is installed just for the load and removed afterwards; leaving it in
    sys.modules would break every later test that imports the real package.
    """
    try:
        import pychunkedgraph.graph.exceptions as exc  # pylint: disable=import-outside-toplevel

        return _load("_readiness_under_test", "app/readiness.py"), exc
    except ImportError:
        pass
    added = [m for m in ("cloudfiles",) if m not in sys.modules]
    if added:
        sys.modules["cloudfiles"] = types.ModuleType("cloudfiles")
        sys.modules["cloudfiles"].CloudFiles = FakeCloudFiles
    exc = _load("_exceptions_under_test", "graph/exceptions.py")
    graph = types.ModuleType("pychunkedgraph.graph")
    graph.exceptions = exc
    saved = {k: sys.modules.get(k) for k in ("pychunkedgraph.graph", "pychunkedgraph.graph.exceptions")}
    sys.modules["pychunkedgraph.graph"] = graph
    sys.modules["pychunkedgraph.graph.exceptions"] = exc
    try:
        module = _load("_readiness_under_test", "app/readiness.py")
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        for name in added:
            sys.modules.pop(name, None)
    return module, exc


readiness, exceptions = _load_readiness()

WS = "gs://ws-bucket/seg"
DYNAMIC = f"{WS}/graphene_meshes/dynamic"


def make_cg(custom_data=None, ws_info=None):
    if custom_data is None:
        custom_data = {"mesh": {"dir": "graphene_meshes"}}
    if ws_info is None:
        ws_info = {"mesh": "graphene_meshes", "mesh_metadata": {"unsharded_mesh_dir": "dynamic"}}
    meta = SimpleNamespace(
        custom_data=custom_data,
        ws_cv=SimpleNamespace(info=ws_info),
        data_source=SimpleNamespace(WATERSHED=WS, EDGES="gs://e/edges", COMPONENTS=None),
        READ_ONLY=custom_data.get("READ_ONLY", False),
    )
    return SimpleNamespace(meta=meta, graph_id="test_graph", version="3.0.0")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    FakeCloudFiles.FILES = {
        f"{WS}/info": b"{}",
        f"{WS}/graphene_meshes/info": b"{}",
        "gs://e/edges/0_0_0": b"",
    }
    FakeCloudFiles.DENY_WRITE = set()
    redis = FakeRedis()
    monkeypatch.setattr(readiness, "CloudFiles", FakeCloudFiles)
    monkeypatch.setattr(readiness, "_redis", lambda: redis)
    monkeypatch.setattr(readiness, "AUTH_URL", "")
    readiness._LOCAL_CACHE.clear()
    return redis


def by_id(report):
    return {c["id"]: c for c in report["checks"]}


def test_healthy_graph_reports_ok_and_leaves_no_probe_objects():
    report = readiness.run_checks("test_graph", lambda t: make_cg())
    checks = by_id(report)
    assert checks["pcg.mesh.dynamic_writable"]["status"] == "ok"
    assert checks["pcg.ws.info_readable"]["status"] == "ok"
    assert checks["pcg.components.readable"]["status"] == "skip"
    assert not [k for k in FakeCloudFiles.FILES if readiness.PROBE_PREFIX in k]
    # Checks come back in onboarding-step order.
    steps = [c["step"] for c in report["checks"]]
    assert steps == sorted(steps, key=readiness.STEPS.index)


def test_unwritable_mesh_bucket_fails_with_remediation():
    FakeCloudFiles.DENY_WRITE = {DYNAMIC}
    report = readiness.run_checks("test_graph", lambda t: make_cg())
    check = by_id(report)["pcg.mesh.dynamic_writable"]
    assert report["status"] == "fail"
    assert check["status"] == "fail" and check["step"] == "infra"
    assert "gs://ws-bucket" in check["remediation"]


def test_missing_mesh_config_warns_and_skips_probe():
    report = readiness.run_checks("test_graph", lambda t: make_cg(custom_data={}))
    checks = by_id(report)
    assert checks["pcg.mesh.configured"]["status"] == "warn"
    assert "pcg.mesh.dynamic_writable" not in checks


def test_unopenable_table_is_reported_not_raised():
    def boom(_):
        raise ValueError("Graph x not supported.")

    report = readiness.run_checks("x", boom)
    assert by_id(report)["pcg.table.usable"]["status"] == "fail"


def test_unmapped_table_fails_auth_check(monkeypatch):
    monkeypatch.setattr(readiness, "AUTH_URL", "global.example.org/auth")
    monkeypatch.setattr(
        readiness.requests, "get", lambda *a, **k: SimpleNamespace(status_code=404, text="")
    )
    report = readiness.run_checks("test_graph", lambda t: make_cg())
    assert by_id(report)["pcg.auth.table_mapped"]["status"] == "fail"
    assert report["auth_server"] == "global.example.org/auth"


def test_get_report_is_cached_until_refresh():
    calls = []

    def get_cg(t):
        calls.append(t)
        return make_cg()

    readiness.get_report("test_graph", get_cg)
    readiness.get_report("test_graph", get_cg)
    assert len(calls) == 1
    readiness.get_report("test_graph", get_cg, refresh=True)
    assert len(calls) == 2


def test_edit_gate_blocks_when_mesh_writes_fail():
    FakeCloudFiles.DENY_WRITE = {DYNAMIC}
    with pytest.raises(exceptions.ServiceUnavailable) as e:
        readiness.check_edit_allowed(make_cg())
    assert e.value.status_code == 503
    assert "Admin action" in str(e.value)


def test_edit_gate_probes_once_then_uses_cache(monkeypatch, env):
    probes = []
    real = readiness._probe_write
    monkeypatch.setattr(readiness, "_probe_write", lambda p: probes.append(p) or real(p))
    cg = make_cg()
    for _ in range(5):
        readiness.check_edit_allowed(cg)
    assert len(probes) == 1

    # A redis outage still does not cost a probe per edit: the process-local cache answers.
    env.down = True
    for _ in range(5):
        readiness.check_edit_allowed(cg)
    assert len(probes) == 1


def test_refresh_after_fix_unblocks_edits():
    FakeCloudFiles.DENY_WRITE = {DYNAMIC}
    cg = make_cg()
    with pytest.raises(exceptions.ServiceUnavailable):
        readiness.check_edit_allowed(cg)

    FakeCloudFiles.DENY_WRITE = set()  # admin adds the IAM binding
    readiness.get_report("test_graph", lambda t: cg, refresh=True)
    readiness.check_edit_allowed(cg)


def test_edit_gate_can_be_bypassed_per_graph():
    FakeCloudFiles.DENY_WRITE = {DYNAMIC}
    cg = make_cg(
        custom_data={
            "mesh": {"dir": "graphene_meshes"},
            "readiness": {"enforce_edit_gate": False},
        }
    )
    readiness.check_edit_allowed(cg)
