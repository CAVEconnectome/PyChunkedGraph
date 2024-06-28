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


def get_l2children(cg, node: np.uint64) -> np.ndarray:
    nodes = np.array([node], dtype=np.uint64)
    layers = cg.get_chunk_layers(nodes)
    assert np.all(layers >= 2), "nodes must be at layers >= 2"
    l2children = []
    while nodes.size:
        children = cg.get_children(nodes, flatten=True)
        layers = cg.get_chunk_layers(children)
        l2children.append(children[layers == 2])
        nodes = children[layers > 2]
    return np.concatenate(l2children)


def sanity_check(cg, new_roots, operation_id):
    """
    Check for duplicates in hierarchy, useful for debugging.
    """
    # print(f"{len(new_roots)} new ids from {operation_id}")
    l2c_d = {}
    for new_root in new_roots:
        l2c_d[new_root] = get_l2children(cg, new_root)
    success = True
    for k, v in l2c_d.items():
        success = success and (len(v) == np.unique(v).size)
        # print(f"{k}: {np.unique(v).size}, {len(v)}")
    if not success:
        raise RuntimeError("Some ids are not valid.")


def sanity_check_single(cg, node, operation_id):
    v = get_l2children(cg, node)
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
