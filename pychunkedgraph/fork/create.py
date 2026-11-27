"""Fork a graph at the newest safe time at or before the request."""

import uuid
from copy import deepcopy
from datetime import datetime

from cloudfiles import CloudFiles
from kvdbclient.exceptions import PreconditionError
from kvdbclient.fork.marker import ForkMarker, Hop, MarkerState
from kvdbclient.safe_time import safe_time

from pychunkedgraph.graph import BackendClientInfo, ChunkedGraph
from pychunkedgraph.graph import get_client_class, get_default_client_info
from pychunkedgraph.graph import get_valid_timestamp
from pychunkedgraph.graph.exceptions import ChunkedGraphError
from pychunkedgraph.graph.meta import ChunkedGraphMeta
from pychunkedgraph.graph.operation import MAX_EDIT_AGE
from pychunkedgraph.meshing.mesh_meta import MeshMeta


def holds_files(path: str) -> bool:
    """Whether the cloud dir at `path` holds any file; the first one listed answers."""
    return next(iter(CloudFiles(path).list()), None) is not None


def _table_exists(client_class: type, table_id: str, config) -> bool:
    """Whether `table_id`'s storage exists, asked of a client opened and closed for it."""
    client = client_class(table_id, config=config)
    try:
        return client._table_exists()
    finally:
        client.close()


def _refuse(source: str, meta: ChunkedGraphMeta, fork_meta: ChunkedGraphMeta) -> None:
    """Raises `PreconditionError` for a source splitting supervoxels in OCDBT, which a fork
    cannot follow, or a fork whose own dynamic dir holds a deleted namesake's meshes."""
    if meta.ocdbt_seg:
        raise PreconditionError(
            f"{source} splits supervoxels in OCDBT, which forks cannot"
        )
    if "dir" in fork_meta.custom_data.get("mesh", {}):
        path = MeshMeta(fork_meta).dynamic_path
        if holds_files(path):
            raise PreconditionError(
                f"{path} already holds meshes; clear it to reuse the name"
            )


def create_fork(
    source: str,
    fork: str,
    at: datetime,
    *,
    wait: bool = False,
    client_info: BackendClientInfo = get_default_client_info(),
) -> ChunkedGraph:
    """Fork `source` into the new table `fork` at the newest safe time at or before `at`.

    An edit that may still write holds that time back: the fork steps below it, or with
    `wait` waits for it to finish. A fork `_refuse` names, or an existing table, raises
    `PreconditionError` before anything is written.
    """
    ClientClass = get_client_class(client_info.TYPE)
    if not ClientClass.opens_forks:
        raise ChunkedGraphError(f"{client_info.TYPE} tables cannot be forked")
    if _table_exists(ClientClass, fork, client_info.CONFIG):
        raise PreconditionError(f"{fork} already exists")
    upstream = ClientClass(source, config=client_info.CONFIG)
    try:
        meta = upstream.read_table_meta()
        config = meta.graph_config._replace(ID=fork, ID_PREFIX="")
        # A copy's mesh block: own dynamic dir, source's dirs read for what it inherits.
        fork_meta = ChunkedGraphMeta(
            config, meta.data_source, deepcopy(meta.custom_data)
        ).for_copied_graph(fork)
        _refuse(source, meta, fork_meta)
        # A source that is itself a fork passes on its chain, read beyond this hop.
        inherited = upstream.marker.chain if upstream.marker else ()
        requested = get_valid_timestamp(at)
        tf = safe_time(upstream, requested, MAX_EDIT_AGE, wait=wait)
        marker = ForkMarker(
            chain=(Hop(source, tf),) + inherited,
            state=MarkerState.READY,
            generation=uuid.uuid4().hex,
            source_max_op_id=int(upstream.get_max_operation_id()),
            requested_at=requested,
        )
        version = upstream.read_table_version()
    finally:
        upstream.close()
    ClientClass.create_fork_table(
        fork, marker, fork_meta, version, config=client_info.CONFIG
    ).close()
    return ChunkedGraph(graph_id=fork, client_info=client_info)
