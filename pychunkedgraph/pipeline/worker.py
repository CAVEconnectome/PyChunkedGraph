"""Generic one-shot batch-worker harness — the container entrypoint skeleton.

A workload calls ``run(make_processor)``: this reads the Indexed-Job env contract,
maps JOB_COMPLETION_INDEX to a batch of scattered chunk coords, and runs each
through the workload's per-chunk processor, returning an exit code for the Job's
podFailurePolicy. No Redis, no yaml — the graph is read from Bigtable by graph id.

``make_processor(cg, layer, env) -> process_one``; ``process_one(coord)`` returns
``"ok" | "done" | "transient" | "fatal"``.
"""

import logging
import os

from ..graph.chunkedgraph import ChunkedGraph
from . import grid
from .exit_codes import FATAL, SUCCESS, TRANSIENT

logger = logging.getLogger(__name__)
NOTE = logging.INFO + 5  # progress level above other libs' INFO so they stay quiet
logging.addLevelName(NOTE, "NOTE")


def layer_bounds(cg, layer: int):
    """(X,Y,Z) chunk grid for a layer; the root layer is a single chunk."""
    if layer == cg.meta.layer_count:
        return (1, 1, 1)
    return cg.meta.layer_chunk_bounds[layer]


def run(make_processor) -> int:
    """Run one batch index for the configured layer; returns a process exit code."""
    logging.basicConfig(level=NOTE)
    env = {
        "graph_id": os.environ["PCG_GRAPH_ID"],
        "layer": int(os.environ["PCG_LAYER"]),
        "seed": int(os.environ["PCG_PERM_SEED"]),
        "batch_size": int(os.environ["PCG_BATCH_SIZE"]),
        "index": int(os.environ["JOB_COMPLETION_INDEX"]),
        "n_threads": int(os.environ.get("PCG_N_THREADS", 1)),
    }
    layer, index = env["layer"], env["index"]

    # One ChunkedGraph per pod, reused for the whole batch: the meta row is read
    # once here (not per chunk) so it never hot-rows.
    cg = ChunkedGraph(graph_id=env["graph_id"])
    process_one = make_processor(cg, layer, env)

    coords = grid.batch_coords(index, layer_bounds(cg, layer), env["seed"], env["batch_size"])
    logger.log(NOTE, f"layer {layer} batch {index}: {len(coords)} chunks")

    fatal = transient = 0
    for coord in coords:
        outcome = process_one(coord)
        if outcome == "fatal":
            fatal += 1
        elif outcome == "transient":
            transient += 1

    logger.log(
        NOTE,
        f"layer {layer} batch {index} done: {len(coords) - fatal - transient} ok, "
        f"{transient} transient, {fatal} fatal",
    )
    # Retry the batch while any chunk is transiently unfinished (done ones skip);
    # only FailIndex once nothing but fatal chunks remain.
    if transient:
        return TRANSIENT
    if fatal:
        return FATAL
    return SUCCESS
