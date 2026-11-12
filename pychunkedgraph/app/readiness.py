# pylint: disable=invalid-name, missing-docstring, broad-exception-caught
"""
Dataset readiness checks.

Each check verifies one onboarding prerequisite for a graph from inside the serving pod, i.e. with
the same identity the service actually runs as. The payload follows the CAVE readiness contract:
checks are grouped by onboarding `step` and carry a `remediation` an admin can act on.

Checks tagged `required_for_edit` also gate edit operations (see `check_edit_allowed`).
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import requests
from cloudfiles import CloudFiles

from pychunkedgraph.graph import exceptions as cg_exceptions

SERVICE = "pychunkedgraph"
STEPS = ("ingest", "meshing", "infra", "auth", "info")
OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

PROBE_PREFIX = ".cave_readiness_probe"
# Shared (redis) TTL: how long a result is trusted across pods.
READINESS_TTL = int(os.environ.get("READINESS_TTL", "600"))
# Per-process TTL in front of redis, so the edit hot path is a dict lookup and a redis outage
# does not turn every edit into a probe write.
LOCAL_TTL = int(os.environ.get("READINESS_LOCAL_TTL", "60"))
DOCS_URL = os.environ.get("READINESS_DOCS_URL", "")
AUTH_URL = os.environ.get("AUTH_URL", os.environ.get("AUTH_URI", ""))
HTTP_TIMEOUT = 10


def _check(
    check_id,
    step,
    status,
    message="",
    remediation="",
    required_for_edit=False,
):
    anchor = check_id.replace(".", "-")
    return {
        "id": check_id,
        "step": step,
        "status": status,
        "message": message,
        "remediation": remediation,
        "docs": f"{DOCS_URL}#{anchor}" if DOCS_URL else "",
        "required_for_edit": required_for_edit,
    }


def _overall(checks):
    statuses = {c["status"] for c in checks}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return OK


def _bucket(path):
    """`gs://bucket/a/b` -> `gs://bucket`; used to word remediations."""
    protocol, _, rest = path.partition("://")
    return f"{protocol}://{rest.split('/', 1)[0]}"


def _probe_write(path):
    """Write then delete a uniquely named object under `path`. Raises on failure."""
    cf = CloudFiles(path)
    name = f"{PROBE_PREFIX}_{uuid.uuid4().hex}"
    cf.put(name, b"cave readiness probe", compress=None, cache_control="no-cache")
    cf.delete(name)


def _exists(path, filename):
    return CloudFiles(path).exists(filename)


def _any_object(path):
    return next(iter(CloudFiles(path).list()), None) is not None


def dynamic_mesh_path(meta):
    """Where remeshing writes; mirrors app/meshing/common.py:_remeshing."""
    mesh_dir = meta.custom_data.get("mesh", {}).get("dir")
    info = meta.ws_cv.info
    unsharded = info.get("mesh_metadata", {}).get("unsharded_mesh_dir")
    if mesh_dir is None or unsharded is None:
        return None
    return os.path.join(meta.data_source.WATERSHED, mesh_dir, unsharded)


# ------------------------------------------------------------------- individual checks


def check_ws_info(meta):
    ws = meta.data_source.WATERSHED
    try:
        if _exists(ws, "info"):
            return _check("pcg.ws.info_readable", "ingest", OK, f"Read {ws}/info")
        return _check(
            "pcg.ws.info_readable",
            "ingest",
            FAIL,
            f"{ws}/info does not exist",
            "Fix data_source.WATERSHED in the graph metadata or upload the watershed info.",
        )
    except Exception as e:
        return _check(
            "pcg.ws.info_readable",
            "ingest",
            FAIL,
            f"Cannot read {ws}/info: {e}",
            f"Grant this service account read access on {_bucket(ws)}.",
        )


def check_source_readable(meta, name):
    path = getattr(meta.data_source, name, None)
    check_id = f"pcg.{name.lower()}.readable"
    if not path:
        return _check(check_id, "ingest", SKIP, f"No {name} source configured")
    try:
        if _any_object(path):
            return _check(check_id, "ingest", OK, f"Listed {path}")
        return _check(
            check_id,
            "ingest",
            WARN,
            f"{path} is empty",
            "Check that ingest wrote to the configured location.",
        )
    except Exception as e:
        return _check(
            check_id,
            "ingest",
            FAIL,
            f"Cannot list {path}: {e}",
            f"Grant this service account read access on {_bucket(path)}.",
        )


def check_mesh(meta):
    """Returns the meshing checks plus the infra write probe for dynamic meshes."""
    mesh = meta.custom_data.get("mesh")
    if not mesh or not mesh.get("dir"):
        return [
            _check(
                "pcg.mesh.configured",
                "meshing",
                WARN,
                "Graph metadata has no mesh config; meshing is not set up",
                "Run the CAVEpipelines meshing setup for this graph.",
            )
        ]

    checks = []
    ws = meta.data_source.WATERSHED
    info = meta.ws_cv.info
    if "mesh_metadata" not in info or "mesh" not in info:
        checks.append(
            _check(
                "pcg.mesh.configured",
                "meshing",
                FAIL,
                f"{ws}/info is missing 'mesh' or 'mesh_metadata'",
                "Re-run the CAVEpipelines meshing setup, which writes these keys.",
            )
        )
        return checks
    checks.append(
        _check("pcg.mesh.configured", "meshing", OK, f"Mesh dir '{mesh['dir']}'")
    )

    sharded = os.path.join(ws, mesh["dir"])
    try:
        status = OK if _exists(sharded, "info") else FAIL
        checks.append(
            _check(
                "pcg.mesh.initial_readable",
                "meshing",
                status,
                f"{sharded}/info {'found' if status == OK else 'missing'}",
                "" if status == OK else "Run the CAVEpipelines meshing setup.",
            )
        )
    except Exception as e:
        checks.append(
            _check(
                "pcg.mesh.initial_readable",
                "meshing",
                FAIL,
                f"Cannot read {sharded}/info: {e}",
                f"Grant this service account read access on {_bucket(sharded)}.",
            )
        )

    checks.append(check_dynamic_mesh_writable(meta))
    return checks


def check_dynamic_mesh_writable(meta):
    path = dynamic_mesh_path(meta)
    if path is None:
        return _check(
            "pcg.mesh.dynamic_writable",
            "infra",
            SKIP,
            "Meshing not configured; nothing to write",
            required_for_edit=True,
        )
    try:
        _probe_write(path)
        return _check(
            "pcg.mesh.dynamic_writable",
            "infra",
            OK,
            f"Wrote and deleted a probe object in {path}",
            required_for_edit=True,
        )
    except Exception as e:
        return _check(
            "pcg.mesh.dynamic_writable",
            "infra",
            FAIL,
            f"This service cannot write meshes to {path}: {e}",
            f"Grant the pychunkedgraph service account write access (object create/delete) on "
            f"{_bucket(path)}, e.g. via the pcg bucket bindings in terraform-cave-private.",
            required_for_edit=True,
        )


def check_auth_mapping(table_id, token):
    if not AUTH_URL:
        return _check(
            "pcg.auth.table_mapped", "auth", WARN, "AUTH_URL is not configured"
        )
    url = f"https://{AUTH_URL}/api/v1/service/{SERVICE}/table/{table_id}/dataset"
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    except Exception as e:
        return _check(
            "pcg.auth.table_mapped",
            "auth",
            FAIL,
            f"Cannot reach auth server {AUTH_URL}: {e}",
        )
    if resp.status_code == 200:
        return _check(
            "pcg.auth.table_mapped",
            "auth",
            OK,
            f"Table mapped to dataset '{resp.text.strip().strip(chr(34))}'",
        )
    if resp.status_code == 404:
        return _check(
            "pcg.auth.table_mapped",
            "auth",
            FAIL,
            f"Table '{table_id}' is not mapped to a dataset on {AUTH_URL}",
            f"Add a service table mapping ({SERVICE}, {table_id}) -> <dataset> in middle_auth.",
        )
    return _check(
        "pcg.auth.table_mapped",
        "auth",
        FAIL,
        f"Auth server returned {resp.status_code} for {url}",
    )


def check_read_only(meta):
    if meta.READ_ONLY:
        return _check(
            "pcg.read_only",
            "info",
            WARN,
            "Graph is marked READ_ONLY; edits are disabled",
            "Unset custom_data.READ_ONLY if this graph should accept edits.",
        )
    return _check("pcg.read_only", "info", OK, "Graph accepts edits")


# ------------------------------------------------------------------- entry points


def _redis():
    from pychunkedgraph.utils.redis import get_redis_connection

    return get_redis_connection()


_LOCAL_CACHE = {}
_LOCAL_LOCK = threading.Lock()


def _cache_get(key):
    """Process-local first, then redis. Returns None on miss or if redis is unavailable."""
    now = time.monotonic()
    with _LOCAL_LOCK:
        hit = _LOCAL_CACHE.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    try:
        cached = _redis().get(key)
    except Exception:
        return None
    if cached is None:
        return None
    value = json.loads(cached)
    with _LOCAL_LOCK:
        _LOCAL_CACHE[key] = (now + LOCAL_TTL, value)
    return value


def _cache_set(key, value):
    with _LOCAL_LOCK:
        _LOCAL_CACHE[key] = (time.monotonic() + LOCAL_TTL, value)
    try:
        _redis().set(key, json.dumps(value), ex=READINESS_TTL)
    except Exception:
        pass


def _report_key(table_id):
    return f"{table_id}:readiness:report"


def _edit_gate_key(graph_id):
    return f"{graph_id}:readiness:edit"


def get_report(table_id, get_cg, token=None, refresh=False):
    """Cached readiness report; `refresh` forces the checks to re-run."""
    if not refresh:
        report = _cache_get(_report_key(table_id))
        if report is not None:
            return report
    report = run_checks(table_id, get_cg, token=token)
    _cache_set(_report_key(table_id), report)
    return report


def run_checks(table_id, get_cg, token=None):
    """Full readiness report for `table_id`. Uncached; prefer `get_report`.

    get_cg: callable(table_id) -> ChunkedGraph; failures there are reported, not raised.
    """
    checks = []
    try:
        cg = get_cg(table_id)
    except Exception as e:
        checks.append(
            _check(
                "pcg.table.usable",
                "ingest",
                FAIL,
                f"Cannot open graph '{table_id}': {e}",
                "Check that ingest created the table and that its version is supported "
                "(or add it to PCG_GRAPH_IDS).",
            )
        )
        checks.append(check_auth_mapping(table_id, token))
        return _report(table_id, checks)

    meta = cg.meta
    checks.append(
        _check("pcg.table.usable", "ingest", OK, f"Opened graph v{cg.version}")
    )
    checks.append(check_ws_info(meta))
    checks.append(check_source_readable(meta, "EDGES"))
    checks.append(check_source_readable(meta, "COMPONENTS"))
    try:
        checks.extend(check_mesh(meta))
    except Exception as e:
        checks.append(
            _check("pcg.mesh.configured", "meshing", FAIL, f"Mesh check failed: {e}")
        )
    checks.append(check_auth_mapping(table_id, token))
    checks.append(check_read_only(meta))
    # Refresh the edit gate so an admin who just fixed IAM unblocks edits by re-running readiness.
    _cache_set(
        _edit_gate_key(cg.graph_id), [c for c in checks if c["required_for_edit"]]
    )
    return _report(table_id, checks)


def _report(table_id, checks):
    order = {step: i for i, step in enumerate(STEPS)}
    checks = sorted(checks, key=lambda c: order.get(c["step"], len(STEPS)))
    return {
        "service": SERVICE,
        "table_id": table_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "auth_server": AUTH_URL,
        "status": _overall(checks),
        "checks": checks,
    }


def check_edit_allowed(cg):
    """Raise ServiceUnavailable if a `required_for_edit` check fails for this graph.

    Normally a dict lookup: results are cached per process (LOCAL_TTL) and in redis
    (READINESS_TTL); the probe write only runs on a miss in both.
    Admins can bypass with custom_data["readiness"]["enforce_edit_gate"] = False.
    """
    if not cg.meta.custom_data.get("readiness", {}).get("enforce_edit_gate", True):
        return

    key = _edit_gate_key(cg.graph_id)
    checks = _cache_get(key)
    if checks is None:
        checks = [check_dynamic_mesh_writable(cg.meta)]
        _cache_set(key, checks)

    failed = [c for c in checks if c["required_for_edit"] and c["status"] == FAIL]
    if failed:
        details = " ".join(f"{c['message']} Admin action: {c['remediation']}" for c in failed)
        raise cg_exceptions.ServiceUnavailable(
            f"Edits are disabled for '{cg.graph_id}' because the service is not fully "
            f"configured. {details}"
        )
