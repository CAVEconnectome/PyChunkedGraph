# pylint: disable=invalid-name, missing-docstring, logging-fstring-interpolation

"""Pubsub worker that updates coarser segmentation mips after an SV split.

Consumes the same edits exchange the mesh worker uses, but binds its own
queue and filters on the `downsample="true"` attribute set by
`publish_edit` when `result.seg_bbox` is populated. For each block the
SV-split touched, acquires the block's lock, runs the in-memory /
per-mip pyramid writer, releases.
"""

import gc
import logging
import pickle
from os import getenv

from messagingclient import MessagingClient

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.downsample import blocks_for_bbox, process_block
from pychunkedgraph.graph.locks import DownsampleBlockLock

PCG_CACHE = {}

INFO_HIGH = 25
logging.basicConfig(
    level=INFO_HIGH,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def callback(payload):
    # Filter by attribute rather than queue binding so all edit-triggered
    # workers can share the same exchange. Split edits set
    # `downsample=true`; merges/undos/redos/rollbacks don't.
    if payload.attributes.get("downsample") != "true":
        return

    data = pickle.loads(payload.data)
    op_id = int(data["operation_id"])
    table_id = payload.attributes["table_id"]
    seg_bboxes = data.get("seg_bboxes")
    if not seg_bboxes:
        return

    try:
        cg = PCG_CACHE[table_id]
    except KeyError:
        cg = ChunkedGraph(graph_id=table_id)
        PCG_CACHE[table_id] = cg

    # Defensive: non-OCDBT graphs have no coarser scales to write to.
    seg_cfg = cg.meta.custom_data.get("seg", {})
    if not seg_cfg.get("ocdbt"):
        logging.log(
            INFO_HIGH,
            f"graph {table_id} not OCDBT-backed; skipping downsample op {op_id}",
        )
        return

    # Each published bbox is one SV split's write region. Collapse the
    # list into the union of blocks touched so we lock/process each
    # block exactly once even if two bboxes share blocks.
    unique_blocks = set()
    for bbs, bbe in seg_bboxes:
        unique_blocks.update(blocks_for_bbox(cg.meta, bbs, bbe))
    block_list = sorted(unique_blocks)

    logging.log(
        INFO_HIGH,
        f"downsampling {len(block_list)} block(s) for op {op_id} graph {table_id}",
    )
    with DownsampleBlockLock(cg, block_list, op_id):
        for block in block_list:
            process_block(cg.meta, block, seg_bboxes)
    logging.log(INFO_HIGH, f"downsample complete op {op_id} graph {table_id}")
    gc.collect()


c = MessagingClient()
downsample_queue = getenv("PYCHUNKEDGRAPH_DOWNSAMPLE_QUEUE")
assert (
    downsample_queue is not None
), "env PYCHUNKEDGRAPH_DOWNSAMPLE_QUEUE not specified."
c.consume(downsample_queue, callback)
