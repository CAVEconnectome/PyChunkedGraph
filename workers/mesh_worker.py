# pylint: disable=invalid-name, missing-docstring, too-many-locals, logging-fstring-interpolation

import gc
import pickle
import logging
from os import getenv

import numpy as np
from messagingclient import MessagingClient

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph import basetypes
from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing.mesh_dir import dynamic

PCG_CACHE = {}

# Own handler + level: messagingclient/grpc configure root logging at import, so
# logging.basicConfig is a no-op and root-level records below WARNING are dropped.
logger = logging.getLogger("mesh_worker")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    )
    logger.addHandler(_handler)


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

    try:
        mesh_meta = cg.meta.custom_data["mesh"]
        mesh_dir = mesh_meta["dir"]
        layer = mesh_meta["max_layer"]
        mip = mesh_meta["mip"]
        err = mesh_meta["max_error"]
        cv_unsharded_mesh_dir = mesh_meta.get("dynamic_mesh_dir", "dynamic")
    except KeyError:
        logger.warning("no mesh metadata for %s; ignoring", table_id)
        return

    mesh_path = dynamic(
        cg.meta.custom_data, cg.meta.data_source.WATERSHED, cv_unsharded_mesh_dir
    )

    logger.info(
        "remeshing %s l2 ids %s; graph %s operation %s",
        l2ids.size,
        list(l2ids),
        table_id,
        op_id,
    )
    meshgen.remeshing(
        cg,
        l2ids,
        stop_layer=layer,
        mip=mip,
        max_err=err,
        cv_sharded_mesh_dir=mesh_dir,
        cv_unsharded_mesh_path=mesh_path,
    )
    logger.info("remeshing complete; graph %s operation %s", table_id, op_id)
    gc.collect()


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE")
assert remesh_queue is not None, "env PYCHUNKEDGRAPH_REMESH_QUEUE not specified."
c.consume(remesh_queue, callback)
