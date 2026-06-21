"""Migration workload (pcgv3): upgrade pcgv2 chunks to pcgv3 in place.

Same per-layer chunk grid as ingest; idempotent (recompute cross-chunk edges and
overwrite), so no per-chunk lock. Run as a module:
``python -m pychunkedgraph.pipeline.migrate``.
"""
