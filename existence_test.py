import os
from datetime import datetime
import numpy as np

from pychunkedgraph.graph import chunkedgraph
from pychunkedgraph.graph import attributes

layer = 2
<<<<<<< HEAD
n_chunks = 10000
n_segments_per_chunk = 200
# timestamp = datetime.datetime.fromtimestamp(1588875769) 
timestamp = datetime.datetime.utcnow()
=======
n_chunks = 100
n_segments_per_chunk = 20
timestamp = datetime.fromtimestamp(1588875769)
>>>>>>> d23d9f32c07dbb58c148961d5f4382a7f6ae4094

cg = chunkedgraph.ChunkedGraph(graph_id="minnie3_v0")

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

<<<<<<< HEAD
rows = cg.client.read_nodes(node_ids=node_ids, end_time=timestamp, 
                            properties=attributes.Hierarchy.Parent)
=======
rows = cg.client.read_nodes(node_ids=node_ids)
>>>>>>> d23d9f32c07dbb58c148961d5f4382a7f6ae4094
valid_node_ids = []
non_valid_node_ids = []
for k in rows.keys():
    if len(rows[k]) > 0:
        valid_node_ids.append(k)
    else:
        non_valid_node_ids.append(k)

# roots = cg.get_roots(valid_node_ids, time_stamp=timestamp)

<<<<<<< HEAD
roots = []
try:
    roots = cg.get_roots(valid_node_ids)
    assert len(roots) == len(valid_node_ids)
    print(f"ALL {len(roots)} have been successful!")
except:
    print("At least one node failed. Checking nodes one by one now")

if len(roots) != len(valid_node_ids):
    log_dict = {}
    success_dict = {}
    for node_id in valid_node_ids:
        try:
            root = cg.get_root(node_id, time_stamp=timestamp)
            print(f"Success: {node_id} from chunk {cg.get_chunk_id(node_id)}")
            success_dict[node_id] = True
        except Exception as e:
            print(f"{node_id} from chunk {cg.get_chunk_id(node_id)} failed with {e}")
            success_dict[node_id] = False

            t_id = node_id

            while t_id is not None:
                last_working_chunk = cg.get_chunk_id(t_id)
                t_id = cg.get_parent(t_id)

            print(f"Failed on layer {cg.get_chunk_layer(last_working_chunk)} in chunk {last_working_chunk}")
            log_dict[node_id] = last_working_chunk
=======

log_dict = {}
success_dict = {}
for node_id in valid_node_ids:
    try:
        root = cg.get_root(node_id)
        print(f"Success: {node_id} from chunk {cg.get_chunk_id(node_id)}")
        success_dict[node_id] = True
    except Exception as e:
        print(f"{node_id} from chunk {cg.get_chunk_id(node_id)} failed with {e}")
        success_dict[node_id] = False

        t_id = node_id

        while t_id is not None:
            last_working_chunk = cg.get_chunk_id(t_id)
            t_id = cg.get_parent(t_id)

        print(
            f"Failed on layer {cg.get_chunk_layer(last_working_chunk)} in chunk {last_working_chunk}"
        )
        log_dict[node_id] = last_working_chunk
>>>>>>> d23d9f32c07dbb58c148961d5f4382a7f6ae4094

