"""Kubernetes-native chunk-batch pipeline: one Indexed Job per layer, no Redis/RQ.

Workload-agnostic core (``grid`` scatter, ``lock`` per-chunk CAS, ``exit_codes`` Job
contract, ``worker`` harness) shared by per-workload subpackages (``ingest``,
``meshing``). Each worker reads its JOB_COMPLETION_INDEX, maps it to a batch of
scattered chunk coords, and processes each chunk. Self-contained and branch-portable
(main + pcgv3). Run a workload as a module, e.g. ``python -m pychunkedgraph.pipeline.ingest``.
"""
