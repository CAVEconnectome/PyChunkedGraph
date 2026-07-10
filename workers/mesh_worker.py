# pylint: disable=invalid-name, missing-docstring, too-many-locals, logging-fstring-interpolation

import gc
import pickle
import logging
from os import getenv

import numpy as np
from messagingclient import MessagingClient

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.utils import basetypes
from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing.mesh_meta import MeshMeta


PCG_CACHE = {}


def callback(payload):
    data = pickle.loads(payload.data)
    op_id = int(data["operation_id"])
    l2ids = np.array(data["new_lvl2_ids"], dtype=basetypes.NODE_ID)
    table_id = payload.attributes["table_id"]
    remesh = payload.attributes["remesh"]

    if remesh == "false":
        return

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
        mm = MeshMeta(cg)
        layer = mm.max_layer
        mip = mm.mip
        err = mm.max_error
        mesh_dir = mm.dir
        mesh_path = mm.dynamic_path
    except KeyError:
        logging.warning(f"No metadata found for {cg.graph_id}; ignoring...")
        return

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
