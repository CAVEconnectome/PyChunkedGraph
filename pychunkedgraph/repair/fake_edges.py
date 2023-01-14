# pylint: disable=protected-access,missing-function-docstring,invalid-name,wrong-import-position

"""
Replay merge operations to check if fake edges need to be added.
"""

import asyncio
from datetime import datetime
from datetime import timedelta
from os import environ
from typing import Optional

# environ["BIGTABLE_PROJECT"] = "<>"
# environ["BIGTABLE_INSTANCE"] = "<>"
# environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path>"

import pandas as pd
from google.api_core.exceptions import ServiceUnavailable

from pychunkedgraph.graph import edits
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.operation import GraphEditOperation
from pychunkedgraph.graph.operation import MergeOperation
from pychunkedgraph.graph.utils.generic import get_bounding_box as get_bbox


def _add_fake_edges(cg: ChunkedGraph, operation_id: int, operation_log: dict) -> bool:
    operation = GraphEditOperation.from_operation_id(
        cg, operation_id, multicut_as_split=False
    )

    result = {}
    if not isinstance(operation, MergeOperation):
        return result

    ts = operation_log["timestamp"]
    parent_ts = ts - timedelta(seconds=0.1)
    override_ts = ts + timedelta(microseconds=(ts.microsecond % 1000) + 10)

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

    edges, fake_edge_rows = edits.check_fake_edges(
        cg,
        atomic_edges=operation.added_edges,
        inactive_edges=inactive_edges,
        time_stamp=override_ts,
        parent_ts=parent_ts,
    )

    if len(fake_edge_rows) == 0:
        return {}

    # cg.client.write(fake_edge_rows)
    result["operation_id"] = operation_id
    result["operation_ts"] = ts
    result["fake_edges"] = edges
    return result


async def wrapper(cg, operation_id, operation_log):
    result = {}
    try:
        result = _add_fake_edges(cg, operation_id, operation_log)
    except (AssertionError, ServiceUnavailable) as exc:
        print(f"{operation_id}: {exc}")
    return result


async def add_fake_edges(
    graph_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    cg = ChunkedGraph(graph_id=graph_id)
    logs = cg.client.read_log_entries(start_time=start_time, end_time=end_time)
    print(f"total logs: {len(logs)}")
    retries = []
    for _id, _log in logs.items():
        retries.append(wrapper(cg, _id, _log))

    results = await asyncio.gather(*retries)
    result = []
    for item in results:
        if not item:
            continue
        result.append(item)
    return result
