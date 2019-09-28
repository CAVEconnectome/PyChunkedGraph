"""
cli for running ingest
"""
import sys
import time
from collections import defaultdict
from itertools import product
from typing import List
from typing import Sequence

import numpy as np
import click
from flask.cli import AppGroup

from .ingestionmanager import IngestionManager
from .ran_ingestion_v2 import INGEST_CHANNEL
from .ran_ingestion_v2 import INGEST_QUEUE
from .ran_ingestion_v2 import ingest_into_chunkedgraph
from .ran_ingestion_v2 import enqueue_atomic_tasks
from .ran_ingestion_v2 import create_parent_chunk
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig

ingest_cli = AppGroup("ingest")

redis_cnxn = get_redis_connection()


@ingest_cli.command("graph")
@click.argument("graph_id", type=str)
# @click.option("--agglomeration", required=True, type=str)
# @click.option("--watershed", required=True, type=str)
# @click.option("--edges", required=True, type=str)
# @click.option("--components", required=True, type=str)
@click.option("--processed", is_flag=True, help="Use processed data to build graph")
# @click.option("--data-size", required=False, nargs=3, type=int)
# @click.option("--chunk-size", required=True, nargs=3, type=int)
# @click.option("--fanout", required=False, type=int)
# @click.option("--gcp-project-id", required=False, type=str)
# @click.option("--bigtable-instance-id", required=False, type=str)
# @click.option("--interval", required=False, type=float)
def ingest_graph(
    graph_id,
    # agglomeration,
    # watershed,
    # edges,
    # components,
    processed,
    # chunk_size,
    # data_size=None,
    # fanout=2,
    # gcp_project_id=None,
    # bigtable_instance_id=None,
    # interval=90.0
):
    agglomeration = "gs://ranl-scratch/190410_FAFB_v02_ws_size_threshold_200/agg"
    watershed = (
        "gs://microns-seunglab/drosophila_v0/ws_190410_FAFB_v02_ws_size_threshold_200"
    )
    edges = "gs://akhilesh-pcg/190410_FAFB_v02/edges"
    components = "gs://akhilesh-pcg/190410_FAFB_v02/components"
    use_raw_edges = not processed
    use_raw_components = not processed

    chunk_size = [256, 256, 512]
    fanout = 2
    gcp_project_id = None
    bigtable_instance_id = None

    data_source = DataSource(
        agglomeration, watershed, edges, components, use_raw_edges, use_raw_components
    )
    graph_config = GraphConfig(graph_id, chunk_size, fanout)
    bigtable_config = BigTableConfig(gcp_project_id, bigtable_instance_id)

    redis_cnxn.flushdb()
    imanager = ingest_into_chunkedgraph(data_source, graph_config, bigtable_config)
    redis_cnxn.set(r_keys.INGESTION_MANAGER, imanager.get_serialized_info(pickled=True))
    enqueue_atomic_tasks(imanager)


def _get_children_coords(
    imanager, layer: int, parent_coords: Sequence[int]
) -> List[np.ndarray]:
    layer_bounds = imanager.chunk_id_bounds / (2 ** (layer - 2))
    layer_bounds = np.ceil(layer_bounds).astype(np.int)
    children_coords = []
    parent_coords = np.array(parent_coords, dtype=int)
    for dcoord in product(*[range(imanager.cg.fan_out)] * 3):
        dcoord = np.array(dcoord, dtype=int)
        child_coords = parent_coords * imanager.cg.fan_out + dcoord
        check_bounds = np.less(child_coords, layer_bounds[:, 1])
        if np.all(check_bounds):
            children_coords.append(child_coords)
    return children_coords


def _parse_results(imanager, parent_hash):
    global redis_cnxn
    zset_name = f"rq:finished:{INGEST_QUEUE}"
    results = redis_cnxn.zrange(zset_name, 0, -1)
    layer_counts_d = defaultdict(int)
    parent_chunks_d = defaultdict(list)  # (layer, x, y, z) as keys
    for chunk_str in results:
        chunk_str = chunk_str.decode("utf-8")
        layer, x, y, z = map(int, chunk_str.split("_"))
        layer_counts_d[layer] += 1
        if layer == imanager.n_layers:
            print("All jobs completed.")
            sys.exit(0)
        layer += 1
        x, y, z = np.array([x, y, z], int) // imanager.cg.fan_out
        parent_job_id = f"{layer}_{'_'.join(map(str, (x, y, z)))}"
        if not redis_cnxn.hget(parent_hash, parent_job_id) is None:
            continue
        parent_chunks_d[(layer, x, y, z)].append(chunk_str)
    return parent_chunks_d, layer_counts_d


def _enqueue_parent_tasks():
    """ 
    Helper to enqueue parent tasks
    Checks job/chunk ids in redis to determine if parent task can be enqueued
    """
    global redis_cnxn
    parent_hash = "rq:enqueued:parents"
    imanager = IngestionManager.from_pickle(redis_cnxn.get(r_keys.INGESTION_MANAGER))
    parent_chunks_d, layer_counts_d = _parse_results(imanager, parent_hash)
    count = 0
    for parent_chunk in parent_chunks_d:
        children_coords = _get_children_coords(
            imanager, parent_chunk[0] - 1, parent_chunk[1:]
        )
        children_results = parent_chunks_d[parent_chunk]
        if not len(children_coords) == len(children_results):
            continue
        job_id = f"{parent_chunk[0]}_{'_'.join(map(str, parent_chunk[1:]))}"
        imanager.task_q.enqueue(
            create_parent_chunk,
            job_id=job_id,
            job_timeout="10m",
            result_ttl=86400,
            args=(imanager.get_serialized_info(), parent_chunk[0], children_coords),
        )
        count += 1
        redis_cnxn.hset(parent_hash, job_id, "")

    layers = range(2, imanager.n_layers)
    status = ", ".join([f"{l}:{layer_counts_d[l]}" for l in layers])
    print(f"Queued {count} parents.")
    print(f"Completed chunks (layer:count)\n{status}")


@ingest_cli.command("parents")
@click.option("--interval", required=True, type=float)
def ingest_parent_chunks(interval):
    """
    This can only be used after running `ingest graph`
    Uses serialzed ingestion manager information stored in redis
    by the `ingest graph` command.
    """
    if not redis_cnxn.get(r_keys.INGESTION_MANAGER):
        click.secho("Run `ingest graph` before using this command.", fg="red")
        sys.exit(1)
    while True:
        _enqueue_parent_tasks()
        time.sleep(interval)


def init_ingest_cmds(app):
    app.cli.add_command(ingest_cli)
