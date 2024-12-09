# pylint: disable=invalid-name, missing-docstring, too-many-locals, logging-fstring-interpolation

import gc
import pickle
import logging
from os import path
from os import getenv

import numpy as np
from messagingclient import MessagingClient

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.utils import basetypes
from pychunkedgraph.meshing import meshgen


PCG_CACHE = {}


def callback(payload):
    data = pickle.loads(payload.data)
    op_id = int(data["operation_id"])
    l2ids = np.array(data["new_lvl2_ids"], dtype=basetypes.NODE_ID)
    table_id = payload.attributes["table_id"]

    try:
        cg = PCG_CACHE[table_id]
    except KeyError:
        cg = ChunkedGraph(graph_id=table_id)
        PCG_CACHE[table_id] = cg

    INFO_HIGH = 25
    logging.basicConfig(
        level=INFO_HIGH,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    try:
        mesh_meta = cg.meta.custom_data["mesh"]
        mesh_dir = mesh_meta["dir"]
        layer = mesh_meta["max_layer"]
        mip = mesh_meta["mip"]
        err = mesh_meta["max_error"]
        cv_unsharded_mesh_dir = mesh_meta.get("dynamic_mesh_dir", "dynamic")
    except KeyError:
        logging.warning(f"No metadata found for {cg.graph_id}; ignoring...")
        return

    mesh_path = path.join(
        cg.meta.data_source.WATERSHED, mesh_dir, cv_unsharded_mesh_dir
    )

    logging.log(INFO_HIGH, f"remeshing {l2ids}; graph {table_id} operation {op_id}.")
    meshgen.remeshing(
        cg,
        l2ids,
        stop_layer=layer,
        mip=mip,
        max_err=err,
        cv_sharded_mesh_dir=mesh_dir,
        cv_unsharded_mesh_path=mesh_path,
    )
    logging.log(INFO_HIGH, f"remeshing complete; graph {table_id} operation {op_id}.")
    gc.collect()


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE")
assert remesh_queue is not None, "env PYCHUNKEDGRAPH_REMESH_QUEUE not specified."
c.consume(remesh_queue, callback)
