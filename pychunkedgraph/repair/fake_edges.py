# pylint: disable=protected-access,missing-function-docstring,invalid-name,wrong-import-position

"""
Replay merge operations to check if fake edges need to be added.
"""

import asyncio
from datetime import datetime
from datetime import timedelta
from os import environ
import multiprocessing as mp
from typing import Optional


# environ["BIGTABLE_PROJECT"] = "<>"
# environ["BIGTABLE_INSTANCE"] = "<>"
# environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path>"

import pandas as pd
from google.api_core.exceptions import ServiceUnavailable
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.graph import edits
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.operation import GraphEditOperation
from pychunkedgraph.graph.operation import MergeOperation
from pychunkedgraph.graph.utils.generic import get_bounding_box as get_bbox


def add_fake_edges_single(cg: ChunkedGraph, operation_id: int, timestamp) -> bool:
    operation = GraphEditOperation.from_operation_id(
        cg, operation_id, multicut_as_split=False
    )

    result = {}
    if not isinstance(operation, MergeOperation):
        return result

    parent_ts = timestamp - timedelta(seconds=0.1)
    override_ts = timestamp + timedelta(
        microseconds=(timestamp.microsecond % 1000) + 10
    )

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

    cg.client.write(fake_edge_rows)
    result["operation_id"] = operation_id
    result["operation_ts"] = timestamp
    result["fake_edges"] = edges
    return result


def wrapper(args):
    graph_id, logs, timestamps = args
    results = []
    cg = ChunkedGraph(graph_id=graph_id)

    for log, timestamp in zip(logs, timestamps):
        result = {}
        try:
            result = add_fake_edges_single(cg, log, timestamp)
        except (AssertionError, ServiceUnavailable) as exc:
            print(f"{log}: {exc}")
        if result:
            results.append(result)
    return results


def add_fake_edges(
    graph_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    cg = ChunkedGraph(graph_id=graph_id)
    logs = cg.client.read_log_entries(start_time=start_time, end_time=end_time)

    operations_ids = list(logs.keys())
    batch_size = len(operations_ids) // mp.cpu_count() + 1

    start = 0
    multi_args = []
    for _ in range(mp.cpu_count()):
        batch = operations_ids[start : start + batch_size]
        start += batch_size
        timestamps = [logs[i]["timestamp"] for i in batch]
        multi_args.append((graph_id, batch, timestamps))

    print(f"total: {len(operations_ids)}")
    print(f"cpu count: {mp.cpu_count()}")
    print(f"batch_size: {batch_size}")

    results = mu.multiprocess_func(wrapper, multi_args)

    results_flat = []
    for result in results:
        results_flat.extend(result)
    return results_flat
