from os import path
from os import getenv

from messagingclient import MessagingClient


def callback(payload):
    import logging
    import numpy as np
    from pychunkedgraph.graph import ChunkedGraph
    from pychunkedgraph.meshing import meshgen

    new_lvl2_ids = np.frombuffer(payload.data, dtype=np.uint64)
    table_id = payload.attributes["table_id"]

    cg = ChunkedGraph(graph_id=table_id)
    mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    unsharded_mesh_path = path.join(
        cg.meta.data_source.WATERSHED, mesh_dir, cv_unsharded_mesh_dir
    )

    mesh_data = cg.meta.custom_data["mesh"]
    layer = (mesh_data["max_layer"],)
    mip = (mesh_data["mip"],)
    err = (mesh_data["max_error"],)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Remeshing {new_lvl2_ids.size} L2 IDs in graph: {table_id}")
    logging.info(f"stop_layer={layer}, mip={mip}, max_err={err}")
    logging.info(f"mesh_dir={mesh_dir}, unsharded_mesh_path={unsharded_mesh_path}")
    meshgen.remeshing(
        cg,
        new_lvl2_ids,
        stop_layer=layer,
        mip=mip,
        max_err=err,
        cv_sharded_mesh_dir=mesh_dir,
        cv_unsharded_mesh_path=unsharded_mesh_path,
    )


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE")
assert remesh_queue is not None, "env PYCHUNKEDGRAPH_REMESH_QUEUE not specified."
c.consume(remesh_queue, callback)
