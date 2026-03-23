import numpy as np
from collections import defaultdict, Counter

from pychunkedgraph.graph import ChunkedGraph, attributes, basetypes
from pychunkedgraph.graph.edges.utils import get_cross_chunk_edges_layer


def inspect_stitch_edges(cg: ChunkedGraph, atomic_edges: np.ndarray):
    """
    Read-only inspection of a batch of stitch edges.
    Shows: edge layers, L2 parents, chunk distribution,
    existing cross-chunk edges on affected L2 nodes,
    and the current hierarchy above them.
    """
    atomic_edges = np.asarray(atomic_edges, dtype=basetypes.NODE_ID)
    svs = np.unique(atomic_edges)
    print(f"edges: {len(atomic_edges)}, unique SVs: {len(svs)}")

    edge_layers = get_cross_chunk_edges_layer(cg.meta, atomic_edges)
    layer_counts = Counter(edge_layers.tolist())
    print(f"\nedge layer distribution:")
    for layer in sorted(layer_counts):
        print(f"  layer {layer}: {layer_counts[layer]} edges")

    parents = cg.get_parents(svs)
    sv_parent = dict(zip(svs.tolist(), parents))
    l2ids = np.unique(parents)
    print(f"\nunique L2 nodes: {len(l2ids)}")

    coords = cg.get_chunk_coordinates_multiple(l2ids)
    chunk_counts = Counter(tuple(c) for c in coords)
    print(f"L2 chunks involved: {len(chunk_counts)}")

    atomic_cx = cg.get_atomic_cross_edges(l2ids)
    print(f"\nexisting AtomicCrossChunkEdge on affected L2 nodes:")
    total_by_layer = defaultdict(int)
    for l2id, layer_edges in atomic_cx.items():
        for layer, edges in layer_edges.items():
            total_by_layer[layer] += len(edges)
    for layer in sorted(total_by_layer):
        print(f"  layer {layer}: {total_by_layer[layer]} atomic cross edges")

    roots = cg.get_roots(l2ids)
    root_layers = cg.get_chunk_layers(roots)
    unique_roots, root_counts = np.unique(roots, return_counts=True)
    print(f"\ncurrent hierarchy: {len(unique_roots)} unique roots, layers {np.unique(root_layers).tolist()}")

    return {
        "edge_layers": edge_layers,
        "sv_parent": sv_parent,
        "l2ids": l2ids,
        "atomic_cx": atomic_cx,
        "roots": dict(zip(l2ids.tolist(), roots.tolist())),
        "layer_counts": layer_counts,
    }


def inspect_l2_cross_edges(cg: ChunkedGraph, l2ids: np.ndarray):
    """Read-only: show AtomicCrossChunkEdge and CrossChunkEdge state for L2 nodes."""
    l2ids = np.asarray(l2ids, dtype=basetypes.NODE_ID)
    atomic_cx = cg.get_atomic_cross_edges(l2ids)
    stored_cx = cg.get_cross_chunk_edges(l2ids, raw_only=True)

    for l2id in l2ids:
        print(f"\nL2 {l2id}  chunk={tuple(cg.get_chunk_coordinates(l2id))}")
        acx = atomic_cx.get(l2id, {})
        print(f"  AtomicCrossChunkEdge: {({l: len(e) for l, e in acx.items()}) or '(none)'}")
        scx = stored_cx.get(l2id, {})
        print(f"  CrossChunkEdge:       {({l: len(e) for l, e in scx.items()}) or '(none)'}")


def inspect_hierarchy(cg: ChunkedGraph, l2ids: np.ndarray, stop_layer: int = None):
    """Read-only: trace the parent chain from L2 up to root."""
    if stop_layer is None:
        stop_layer = cg.meta.layer_count
    l2ids = np.asarray(l2ids, dtype=basetypes.NODE_ID)

    for l2id in l2ids:
        print(f"\n--- L2 {l2id} ---")
        node = l2id
        while True:
            layer = cg.get_chunk_layer(node)
            coord = tuple(cg.get_chunk_coordinates(node))
            cx = cg.get_cross_chunk_edges([node], raw_only=True).get(node, {})
            cx_summary = {l: len(e) for l, e in cx.items()}
            print(f"  L{layer} {node} @ {coord}  cx_edges={cx_summary}")
            if layer >= stop_layer:
                break
            parent = cg.get_parent(node)
            if parent is None or parent == 0:
                print(f"  (no parent)")
                break
            node = parent
