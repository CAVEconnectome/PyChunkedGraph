"""Ingest workload for the chunk-batch pipeline.

Builds L2 atomic-edge chunks and parent-layer agglomerations, each under a
per-chunk Bigtable lock. Run as a module: ``python -m pychunkedgraph.pipeline.ingest``.
"""
