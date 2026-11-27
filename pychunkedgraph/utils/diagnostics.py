"""One store for the artifacts that help debug a graph or make it faster.

Each kind of artifact lives under
`{parent of the mesh dir}/graphene_diagnostics/{graph_id}/{kind}`, a sibling of the
mesh dir: remeshing writes there, so it is the one location a server is sure to write,
which a watershed path repointed elsewhere need not be.
"""

from typing import Optional

from pychunkedgraph import get_logger

logger = get_logger(__name__)

ROOT = "graphene_diagnostics"


def _mesh_parent(cg) -> Optional[str]:
    """The path holding the graph's mesh dir; None when it cannot be told.

    A mesh dir stored as a full path stands as it is; a bare name sits under the
    watershed path.
    """
    mesh_dir = str(cg.meta.custom_data.get("mesh", {}).get("dir") or "").rstrip("/")
    if "://" in mesh_dir:
        return mesh_dir.rsplit("/", 1)[0]
    watershed = getattr(cg.meta.data_source, "WATERSHED", None)
    if not watershed:
        return None
    return "/".join([watershed.rstrip("/"), *mesh_dir.split("/")[:-1]])


def store_dir(cg, kind: str) -> Optional[str]:
    """The directory of one kind of artifact for this graph; None with no mesh parent."""
    parent = _mesh_parent(cg)
    if parent is None:
        return None
    return f"{parent}/{ROOT}/{cg.graph_id}/{kind}"


def put_json(cg, kind: str, name: str, payload: dict[str, object]) -> Optional[str]:
    """Write one JSON object into the store. Returns its URL or None; never raises."""
    try:
        base = store_dir(cg, kind)
        if base is None:
            return None
        from cloudfiles import CloudFiles

        CloudFiles(base).put_json(name, payload)
        return f"{base}/{name}"
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.warning(f"diagnostics write failed for {kind}/{name}: {err}")
        return None


def get_json(cg, kind: str, name: str) -> Optional[dict[str, object]]:
    """Read one JSON object from the store; None when missing or unreadable."""
    try:
        base = store_dir(cg, kind)
        if base is None:
            return None
        from cloudfiles import CloudFiles

        return CloudFiles(base).get_json(name)
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.warning(f"diagnostics read failed for {kind}/{name}: {err}")
        return None
