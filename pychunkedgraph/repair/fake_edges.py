# pylint: disable=protected-access,missing-function-docstring,invalid-name,wrong-import-position

"""
Replay merge operations to check if fake edges need to be added.
"""

from datetime import datetime
from datetime import timedelta
from os import environ
from typing import Optional

# environ["BIGTABLE_PROJECT"] = "<>"
# environ["BIGTABLE_INSTANCE"] = "<>"
# environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path>"

from pychunkedgraph.graph import edits
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.operation import GraphEditOperation
from pychunkedgraph.graph.operation import MergeOperation
from pychunkedgraph.graph.utils.generic import get_bounding_box as get_bbox


def _add_fake_edges(cg: ChunkedGraph, operation_id: int, operation_log: dict) -> bool:
    operation = GraphEditOperation.from_operation_id(
        cg, operation_id, multicut_as_split=False
    )

    if not isinstance(operation, MergeOperation):
        return False

    ts = operation_log["timestamp"]
    parent_ts = ts - timedelta(seconds=0.1)
    override_ts = (ts + timedelta(microseconds=(ts.microsecond % 1000) + 10),)

    root_ids = set(
        cg.get_roots(
            operation.added_edges.ravel(), assert_roots=True, time_stamp=parent_ts
        )
    )

    bbox = get_bbox(
        operation.source_coords, operation.sink_coords, operation.bbox_offset
    )
    edges = cg.get_subgraph(
        root_ids,
        bbox=bbox,
        bbox_is_coordinate=True,
        edges_only=True,
    )

    inactive_edges = edits.merge_preprocess(
        cg,
        subgraph_edges=edges,
        supervoxels=operation.added_edges.ravel(),
        parent_ts=parent_ts,
    )

    _, fake_edge_rows = edits.check_fake_edges(
        cg,
        atomic_edges=operation.added_edges,
        inactive_edges=inactive_edges,
        time_stamp=override_ts,
        parent_ts=parent_ts,
    )

    cg.client.write(fake_edge_rows)
    return len(fake_edge_rows) > 0


def add_fake_edges(
    graph_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    cg = ChunkedGraph(graph_id=graph_id)
    logs = cg.client.read_log_entries(start_time=start_time, end_time=end_time)
    for _id, _log in logs.items():
        _add_fake_edges(cg, _id, _log)
