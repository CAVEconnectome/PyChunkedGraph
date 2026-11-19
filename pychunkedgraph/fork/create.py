"""Fork a graph at the newest safe time at or before the request."""

import uuid
from copy import deepcopy
from datetime import datetime

from kvdbclient.fork.marker import ForkMarker, Hop, MarkerState
from kvdbclient.safe_time import safe_time

from pychunkedgraph.graph import BackendClientInfo, ChunkedGraph
from pychunkedgraph.graph import get_client_class, get_default_client_info
from pychunkedgraph.graph import get_valid_timestamp
from pychunkedgraph.graph.exceptions import ChunkedGraphError
from pychunkedgraph.graph.meta import ChunkedGraphMeta
from pychunkedgraph.graph.operation import MAX_EDIT_AGE


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
    `wait` waits for it to finish.
    """
    ClientClass = get_client_class(client_info.TYPE)
    if not ClientClass.opens_forks:
        raise ChunkedGraphError(f"{client_info.TYPE} tables cannot be forked")
    upstream = ClientClass(source, config=client_info.CONFIG)
    meta = upstream.read_table_meta()
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
    config = meta.graph_config._replace(ID=fork, ID_PREFIX="")
    fork_meta = ChunkedGraphMeta(config, meta.data_source, deepcopy(meta.custom_data))
    version = upstream.read_table_version()
    upstream.close()
    ClientClass.create_fork_table(
        fork, marker, fork_meta, version, config=client_info.CONFIG
    ).close()
    return ChunkedGraph(graph_id=fork, client_info=client_info)
