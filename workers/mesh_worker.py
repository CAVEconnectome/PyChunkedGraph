# pylint: disable=invalid-name, missing-docstring, too-many-locals, logging-fstring-interpolation

import gc
import pickle
import logging
from os import getenv

import numpy as np
from cloudfiles import CloudFiles
from messagingclient import MessagingClient

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph import basetypes
from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing.mesh_meta import MeshMeta

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
        mm = MeshMeta(cg)
        layer = mm.max_layer
        mip = mm.mip
        err = mm.max_error
        mesh_dir = mm.dir
        mesh_path = mm.dynamic_path
    except KeyError:
        logger.warning("no mesh metadata for %s; ignoring", table_id)
        return

    if not CloudFiles(cg.meta.data_source.WATERSHED).exists(f"{mesh_dir}/info"):
        logger.warning("no meshes at %s for %s; ignoring", mesh_dir, table_id)
        return

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
        cv_unsharded_mesh_path=mesh_path,
    )
    logger.info("remeshing complete; graph %s operation %s", table_id, op_id)
    gc.collect()


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE")
assert remesh_queue is not None, "env PYCHUNKEDGRAPH_REMESH_QUEUE not specified."
c.consume(remesh_queue, callback)
