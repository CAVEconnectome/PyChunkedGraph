"""Per-workload pipeline workers (ingest, meshing, migrate) for the k8s Indexed-Job
pipeline, built on ``cave-pipeline``'s chunk-distribution core (the grid scatter,
worker harness, per-chunk lock, exit-code contract, and ``run_and_exit``). Each
workload is a module: ``python -m pychunkedgraph.pipeline.<workload>``.

This package supplies only the PyChunkedGraph-specific harness glue — a ChunkedGraph
``context_factory`` and the per-layer ``bounds_fn`` — injected into the shared harness.
"""

from ..graph.chunkedgraph import ChunkedGraph


def cg_factory(env):
    """The harness ``context_factory``: one ChunkedGraph per pod (meta read once)."""
    return ChunkedGraph(graph_id=env["graph_id"])


def layer_bounds(cg, layer: int):
    """(X,Y,Z) chunk grid for a layer; the root layer is a single chunk."""
    if layer == cg.meta.layer_count:
        return (1, 1, 1)
    return cg.meta.layer_chunk_bounds[layer]
