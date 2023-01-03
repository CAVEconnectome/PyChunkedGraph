import os
from datetime import datetime
import numpy as np

from pychunkedgraph.graph import chunkedgraph
from pychunkedgraph.graph import attributes

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  "/home/svenmd/.cloudvolume/secrets/google-secret.json"

layer = 2
n_chunks = 1000
n_segments_per_chunk = 200
# timestamp = datetime.datetime.fromtimestamp(1588875769) 
timestamp = datetime.utcnow()

cg = chunkedgraph.ChunkedGraph(graph_id="pinky_nf_v2")

np.random.seed(42)

node_ids = []
for _ in range(n_chunks):
    c_x = np.random.randint(0, cg.meta.layer_chunk_bounds[layer][0])
    c_y = np.random.randint(0, cg.meta.layer_chunk_bounds[layer][1])
    c_z = np.random.randint(0, cg.meta.layer_chunk_bounds[layer][2])

    chunk_id = cg.get_chunk_id(layer=layer, x=c_x, y=c_y, z=c_z)

    max_segment_id = cg.get_segment_id(cg.id_client.get_max_node_id(chunk_id))

    if max_segment_id < 10:
        continue

    segment_ids = np.random.randint(1, max_segment_id, n_segments_per_chunk)

    for segment_id in segment_ids:
        node_ids.append(cg.get_node_id(np.uint64(segment_id), np.uint64(chunk_id)))

rows = cg.client.read_nodes(node_ids=node_ids, end_time=timestamp, 
                            properties=attributes.Hierarchy.Parent)
valid_node_ids = []
non_valid_node_ids = []
for k in rows.keys():
    if len(rows[k]) > 0:
        valid_node_ids.append(k)
    else:
        non_valid_node_ids.append(k)

cc_edges = cg.get_atomic_cross_edges(valid_node_ids)
cc_ids = np.unique(np.concatenate([np.concatenate(list(v.values())) for v in list(cc_edges.values()) if len(v.values())]))

roots = cg.get_roots(cc_ids)
root_dict = dict(zip(cc_ids, roots)) 
root_dict_vec = np.vectorize(root_dict.get)

for k in cc_edges:
    if len(cc_edges[k]) == 0:
        continue
    local_ids = np.unique(np.concatenate(list(cc_edges[k].values())))
    
    assert len(np.unique(root_dict_vec(local_ids)))