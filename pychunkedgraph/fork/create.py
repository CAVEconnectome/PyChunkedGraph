"""Fork a graph at the requested time, stepped back to where every edit has landed."""

import uuid
from copy import deepcopy
from datetime import datetime

from kvdbclient.fork.client import ForkClient, create_fork_table
from kvdbclient.fork.marker import ForkMarker, Hop, MarkerState

from pychunkedgraph.graph import BackendClientInfo, ChunkedGraph
from pychunkedgraph.graph import get_default_client_info, get_valid_timestamp
from pychunkedgraph.graph.meta import ChunkedGraphMeta
from pychunkedgraph.graph.operation import MAX_EDIT_AGE


def create_fork(
    source: str,
    fork: str,
    at: datetime,
    *,
    client_info: BackendClientInfo = get_default_client_info(),
) -> ChunkedGraph:
    """Fork `source` into the new table `fork` at `at`, or `MAX_EDIT_AGE` ago if later."""
    upstream = ForkClient(source, config=client_info.CONFIG)
    meta = upstream.read_table_meta()
    # A source that is itself a fork passes on its chain, read beyond this hop.
    inherited = upstream.marker.chain if upstream.marker else ()
    requested = get_valid_timestamp(at)
    tf = min(requested, get_valid_timestamp(None) - MAX_EDIT_AGE)
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
    create_fork_table(
        fork, marker, fork_meta, version, config=client_info.CONFIG
    ).close()
    return ChunkedGraph(graph_id=fork, client_info=client_info)
