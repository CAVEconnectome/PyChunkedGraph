from os import path
from os import getenv

from messagingclient import MessagingClient


def callback(payload):
    import gc
    import pickle
    import logging
    import numpy as np
    from pychunkedgraph.graph import ChunkedGraph
    from pychunkedgraph.graph.utils import basetypes
    from pychunkedgraph.meshing import meshgen

    data = pickle.loads(payload.data)
    lvl2_ids = np.array(data["new_lvl2_ids"], dtype=basetypes.NODE_ID)
    table_id = payload.attributes["table_id"]

    cg = ChunkedGraph(graph_id=table_id)
    mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    mesh_path = path.join(
        cg.meta.data_source.WATERSHED, mesh_dir, cv_unsharded_mesh_dir
    )

    try:
        mesh_data = cg.meta.custom_data["mesh"]
        layer = mesh_data["max_layer"]
        mip = mesh_data["mip"]
        err = mesh_data["max_error"]
    except KeyError:
        return

    INFO_PRIORITY = 25
    logging.basicConfig(
        level=INFO_PRIORITY,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.log(INFO_PRIORITY, f"Remeshing {lvl2_ids} L2 IDs in graph {table_id}")
    logging.log(INFO_PRIORITY, f"stop_layer={layer}, mip={mip}, max_err={err}")
    logging.log(INFO_PRIORITY, f"mesh_dir={mesh_dir}, unsharded_mesh_path={mesh_path}")
    meshgen.remeshing(
        cg,
        lvl2_ids,
        stop_layer=layer,
        mip=mip,
        max_err=err,
        cv_sharded_mesh_dir=mesh_dir,
        cv_unsharded_mesh_path=mesh_path,
    )
    logging.log(INFO_PRIORITY, "Remeshing complete.")
    gc.collect()


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE")
assert remesh_queue is not None, "env PYCHUNKEDGRAPH_REMESH_QUEUE not specified."
c.consume(remesh_queue, callback)
