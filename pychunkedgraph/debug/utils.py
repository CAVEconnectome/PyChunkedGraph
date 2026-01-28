# pylint: disable=invalid-name, missing-docstring, bare-except, unidiomatic-typecheck

import numpy as np

from pychunkedgraph.graph.meta import ChunkedGraphMeta, GraphConfig


def print_attrs(d):
    for k, v in d.items():
        try:
            print(k.key)
        except:
            print(k)
        try:
            print(v[:2], "...") if type(v) is np.ndarray and len(v) > 2 else print(v)
        except:
            print(v)


def print_node(cg, node: np.uint64, indent: int = 0, stop_layer: int = 2) -> None:
    children = cg.get_children(node)
    print(f"{' ' * indent}{node}[{len(children)}]")
    if cg.get_chunk_layer(node) <= stop_layer:
        return
    for child in children:
        print_node(cg, child, indent=indent + 4, stop_layer=stop_layer)


def sanity_check(cg, new_roots, operation_id):
    """
    Check for duplicates in hierarchy, useful for debugging.
    """
    # print(f"{len(new_roots)} new ids from {operation_id}")
    l2c_d = {}
    for new_root in new_roots:
        l2c_d[new_root] = cg.get_l2children([new_root])
    success = True
    for k, v in l2c_d.items():
        success = success and (len(v) == np.unique(v).size)
        # print(f"{k}: {np.unique(v).size}, {len(v)}")
    if not success:
        raise RuntimeError(f"{operation_id}: some ids are not valid.")


def sanity_check_single(cg, node, operation_id):
    v = cg.get_l2children([node])
    msg = f"invalid node {node}:"
    msg += f" found {len(v)} l2 ids, must be {np.unique(v).size}"
    assert np.unique(v).size == len(v), f"{msg}, from {operation_id}."
    return v


def update_graph_id(cg, new_graph_id:str):
    old_gc = cg.meta.graph_config._asdict()
    old_gc["ID"] = new_graph_id
    new_gc = GraphConfig(**old_gc)
    new_meta = ChunkedGraphMeta(new_gc, cg.meta.data_source, cg.meta.custom_data)
    cg.update_meta(new_meta, overwrite=True)


def get_random_l1_ids(cg, n_chunks=100, n_per_chunk=10, seed=None):
    """Generate random layer 1 IDs from different chunks."""
    if seed:
        np.random.seed(seed)
    bounds = cg.meta.layer_chunk_bounds[2]
    ids = []
    for _ in range(n_chunks):
        cx, cy, cz = [np.random.randint(0, b) for b in bounds]
        chunk_id = cg.get_chunk_id(layer=2, x=cx, y=cy, z=cz)
        max_seg = cg.get_segment_id(cg.id_client.get_max_node_id(chunk_id))
        if max_seg < 2:
            continue
        for seg in np.random.randint(1, max_seg + 1, n_per_chunk):
            ids.append(cg.get_node_id(np.uint64(seg), np.uint64(chunk_id)))
    return np.array(ids, dtype=np.uint64)
