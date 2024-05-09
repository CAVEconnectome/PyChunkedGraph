# pylint: disable=invalid-name, missing-docstring, bare-except, unidiomatic-typecheck

import numpy as np


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
    print(f"{len(new_roots)} new roots from {operation_id}")
    l2c_d = {}
    for new_root in new_roots:
        l2c_d[new_root] = get_l2children(cg, new_root)
    for k, v in l2c_d.items():
        if len(v) == np.unique(v).size:
            print(f"no duplicates in {k}")
        else:
            raise ValueError(f"duplicates in {k}: {len(v)} {np.unique(v).size}")
