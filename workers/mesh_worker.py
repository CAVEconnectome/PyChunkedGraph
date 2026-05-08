from os import getenv

from messagingclient import MessagingClient
from google.cloud import bigtable
from google.auth import credentials
from google.auth import default as default_creds
from google.cloud import bigtable

class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass

def get_bigtable_client(project_id, emulate=False):
    if emulate:
        creds = DoNothingCreds()
    elif project_id is not None:
        creds, _ = default_creds()
    else:
        creds, project_id = default_creds()
    client = bigtable.Client(admin=True, project=project_id, credentials=creds)
    return client


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

    project = getenv("BIGTABLE_PROJECT", "neuromancer-seung-import")
    instance = getenv("BIGTABLE_INSTANCE", "pychunkedgraph")
    client = get_bigtable_client(project_id=project)
    cg=ChunkedGraph(
        table_id, instance_id=instance, client=client
    )

    meshgen.remeshing(cg, new_lvl2_ids, stop_layer=layer, mip=mip, max_err=err)
    logging.info("Remeshing complete.")
    gc.collect()


c = MessagingClient()
remesh_queue = getenv("PYCHUNKEDGRAPH_REMESH_QUEUE", "test")
c.consume(remesh_queue, callback)
