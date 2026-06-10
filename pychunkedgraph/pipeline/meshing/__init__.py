"""Meshing workload for the chunk-batch pipeline.

One-shot mesh-metadata setup, then per-layer mesh generation (L2 marching cubes,
L>2 sharded stitching), idempotent (overwrites shards), no per-chunk lock. Run as a
module: ``python -m pychunkedgraph.pipeline.meshing``.
"""
