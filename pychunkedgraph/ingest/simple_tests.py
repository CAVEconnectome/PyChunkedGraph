# pylint: disable=invalid-name, missing-function-docstring, broad-exception-caught

"""
Some sanity tests to ensure chunkedgraph was created properly.
"""

from datetime import datetime, timezone
import numpy as np

from pychunkedgraph.graph import attributes, ChunkedGraph


def family(cg: ChunkedGraph):
    np.random.seed(42)
    n_chunks = 100
    n_segments_per_chunk = 200
    timestamp = datetime.now(timezone.utc)

    node_ids = []
    for layer in range(2, cg.meta.layer_count - 1):
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
                node_ids.append(
                    cg.get_node_id(np.uint64(segment_id), np.uint64(chunk_id))
                )

    rows = cg.client.read_nodes(
        node_ids=node_ids, end_time=timestamp, properties=attributes.Hierarchy.Parent
    )
    valid_node_ids = []
    non_valid_node_ids = []
    for k in rows.keys():
        if len(rows[k]) > 0:
            valid_node_ids.append(k)
        else:
            non_valid_node_ids.append(k)

    parents = cg.get_parents(valid_node_ids, time_stamp=timestamp)
    children_dict = cg.get_children(parents)
    for child, parent in zip(valid_node_ids, parents):
        assert child in children_dict[parent]
    print("success")


def existence(cg: ChunkedGraph):
    np.random.seed(42)
    layer = 2
    n_chunks = 100
    n_segments_per_chunk = 200
    timestamp = datetime.now(timezone.utc)
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

    rows = cg.client.read_nodes(
        node_ids=node_ids, end_time=timestamp, properties=attributes.Hierarchy.Parent
    )
    valid_node_ids = []
    non_valid_node_ids = []
    for k in rows.keys():
        if len(rows[k]) > 0:
            valid_node_ids.append(k)
        else:
            non_valid_node_ids.append(k)

    roots = []
    try:
        roots = cg.get_roots(valid_node_ids)
        assert len(roots) == len(valid_node_ids)
        print("success")
    except Exception as e:
        print(f"Something went wrong: {e}")
        print("At least one node failed. Checking nodes one by one:")

    if len(roots) != len(valid_node_ids):
        log_dict = {}
        success_dict = {}
        for node_id in valid_node_ids:
            try:
                _ = cg.get_root(node_id, time_stamp=timestamp)
                print(f"Success: {node_id} from chunk {cg.get_chunk_id(node_id)}")
                success_dict[node_id] = True
            except Exception as e:
                print(f"{node_id} - chunk {cg.get_chunk_id(node_id)} failed: {e}")
                success_dict[node_id] = False
                t_id = node_id
                while t_id is not None:
                    last_working_chunk = cg.get_chunk_id(t_id)
                    t_id = cg.get_parent(t_id)

                layer = cg.get_chunk_layer(last_working_chunk)
                print(f"Failed on layer {layer} in chunk {last_working_chunk}")
                log_dict[node_id] = last_working_chunk


def cross_edges(cg: ChunkedGraph):
    np.random.seed(42)
    layer = 2
    n_chunks = 10
    n_segments_per_chunk = 200
    timestamp = datetime.now(timezone.utc)
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

    rows = cg.client.read_nodes(
        node_ids=node_ids, end_time=timestamp, properties=attributes.Hierarchy.Parent
    )
    valid_node_ids = []
    non_valid_node_ids = []
    for k in rows.keys():
        if len(rows[k]) > 0:
            valid_node_ids.append(k)
        else:
            non_valid_node_ids.append(k)

    cc_edges = cg.get_atomic_cross_edges(valid_node_ids)
    cc_ids = np.unique(
        np.concatenate(
            [
                np.concatenate(list(v.values()))
                for v in list(cc_edges.values())
                if len(v.values())
            ]
        )
    )

    roots = cg.get_roots(cc_ids)
    root_dict = dict(zip(cc_ids, roots))
    root_dict_vec = np.vectorize(root_dict.get)

    for k in cc_edges:
        if len(cc_edges[k]) == 0:
            continue
        local_ids = np.unique(np.concatenate(list(cc_edges[k].values())))
        assert len(np.unique(root_dict_vec(local_ids)))
    print("success")


def run_all(cg: ChunkedGraph):
    print("Running family tests:")
    family(cg)

    print("\nRunning existence tests:")
    existence(cg)

    print("\nRunning cross_edges tests:")
    cross_edges(cg)
