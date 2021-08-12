from os import getenv

from messagingclient import MessagingClient


def callback(payload):
    import numpy as np
    from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
    from pychunkedgraph.meshing import meshgen

    new_lvl2_ids = np.frombuffer(payload.data, dtype=np.uint64)
    table_id = payload.attributes["table_id"]

    cg = ChunkedGraph(table_id)
    meshgen.remeshing(
        cg, new_lvl2_ids, stop_layer=4, mesh_path=None, mip=1, max_err=320
    )


c = MessagingClient()
remesh_topic = getenv("PYCHUNKEDGRAPH_REMESH_TOPIC", "test")
c.consume(remesh_topic, callback)
