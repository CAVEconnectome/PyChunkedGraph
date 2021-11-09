from os import getenv

from messagingclient import MessagingClient


def callback(payload):
    import gc
    import logging
    import pickle
    import numpy as np
    from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
    from pychunkedgraph.backend.chunkedgraph_utils import basetypes
    from pychunkedgraph.meshing import meshgen

    data = pickle.loads(payload.data)
    new_lvl2_ids = np.array(data["new_lvl2_ids"], dtype=basetypes.NODE_ID)

    table_id = payload.attributes["table_id"]
    layer = 4
    mip = 1
    err = 320

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Remeshing {new_lvl2_ids.size} L2 IDs in graph {table_id}")
    logging.info(f"stop_layer={layer}, mip={mip}, max_err={err}")

    cg = ChunkedGraph(table_id)
    meshgen.remeshing(cg, new_lvl2_ids, stop_layer=layer, mip=mip, max_err=err)
    logging.info("Remeshing complete.")
    gc.collect()


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE", "test")
c.consume(remesh_queue, callback)
