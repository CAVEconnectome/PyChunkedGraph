# pylint: disable=invalid-name, missing-docstring

"""
Guardrails for queries whose memory footprint scales with the volume of the
requested bounding box rather than with the size of the segment.

`cg.get_subgraph(...)` resolves the level 2 nodes inside the box and then calls
`get_l2_agglomerations`, which reads the edges of *every* level 2 chunk those
nodes live in -- `ChunkedGraph.read_chunk_edges` fetches whole chunk files, not
just the edges of the requested supervoxels. A generous bounding box therefore
pulls in every supervoxel and edge stored in that volume and can exhaust the
pod's memory even when the requested segment barely touches it.

Because the dominant term depends only on the box, it can be estimated before
any data is read, which is what this module does. These limits are meant for
user facing requests; edit operations and ingest deliberately do not go through
them, so the check is applied by the app layer rather than by the graph itself.
"""

from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np

from . import exceptions as cg_exceptions
from .chunks.utils import normalize_bounding_box


DEFAULT_MAX_BYTES = 5 * 1024**3

# Fallback calibration, used when a dataset has no measured
# `BYTES_PER_L2_CHUNK`. Chunk sizes and resolutions differ between datasets, so
# a per-chunk constant does not transfer; what does transfer reasonably well is
# the cost per unit of *physical* volume, since supervoxel and edge density are
# a property of the EM segmentation rather than of the chunking. The figure
# below is derived for minnie65: ~150 edges per cubic micron at ~28 bytes per
# edge (two node ids, an affinity and an area), times ~3 for the copies made
# while concatenating chunk edges and splitting them into in/out/cross sets.
# It is a rough upper-middle estimate; measure and pin per dataset when a
# dataset turns out to be denser or sparser than that.
DEFAULT_BYTES_PER_CUBIC_MICRON = 12_000


def bytes_per_l2_chunk(meta, limits: Dict) -> float:
    """Memory a single level 2 chunk of edges is expected to cost.

    Uses the dataset's measured `BYTES_PER_L2_CHUNK` when configured, otherwise
    scales `BYTES_PER_CUBIC_MICRON` by the physical volume of a chunk so that
    one calibration applies to any chunk size and resolution.
    """
    measured = limits.get("BYTES_PER_L2_CHUNK")
    if measured:
        return float(measured)
    chunk_nm = np.array(meta.graph_config.CHUNK_SIZE, dtype=float) * np.array(
        meta.resolution, dtype=float
    )
    chunk_um3 = float(np.prod(chunk_nm)) / 1e9
    per_um3 = limits.get("BYTES_PER_CUBIC_MICRON", DEFAULT_BYTES_PER_CUBIC_MICRON)
    return chunk_um3 * float(per_um3)


def level2_chunk_count(meta, bounding_box: Optional[Sequence[Sequence[int]]]) -> int:
    """Number of level 2 chunks spanned by a bounding box in voxel coordinates.

    `None` is treated as the whole dataset, which is what an omitted bounding
    box actually asks for.
    """
    chunk_bounds = np.array(meta.layer_chunk_bounds[2], dtype=int)
    if bounding_box is None:
        bbox = np.array([[0, 0, 0], chunk_bounds], dtype=int)
    else:
        bbox = normalize_bounding_box(meta, np.array(bounding_box, dtype=int), True)
    lower = np.clip(bbox[0], 0, chunk_bounds)
    upper = np.clip(bbox[1], 0, chunk_bounds)
    return int(np.prod(np.maximum(upper - lower, 1)))


def _suggested_box(meta, max_chunks: int) -> Sequence[int]:
    """A roughly cubic box, in voxels, that fits within `max_chunks`.

    Chunks are usually anisotropic, so a cube of chunks would be a very
    elongated region; this sizes each axis in physical space instead and then
    rounds down to whole chunks, which is the granularity that actually counts.
    """
    chunk_size = np.array(meta.graph_config.CHUNK_SIZE, dtype=int)
    chunk_nm = chunk_size * np.array(meta.resolution, dtype=float)
    side_nm = (max_chunks * float(np.prod(chunk_nm))) ** (1 / 3)
    chunks_per_axis = np.maximum(np.floor(side_nm / chunk_nm), 1).astype(int)
    # clamping a thin axis up to one chunk can push the total over the budget
    while np.prod(chunks_per_axis) > max_chunks and np.any(chunks_per_axis > 1):
        chunks_per_axis[np.argmax(chunks_per_axis)] -= 1
    return (chunks_per_axis * chunk_size).tolist()


def check_subgraph_bounds(
    cg,
    bounding_box: Optional[Sequence[Sequence[int]]],
    limits: Optional[Dict],
) -> None:
    """Reject a subgraph request whose bounding box is too expensive to serve.

    `limits` may set `MAX_BYTES`, the memory a single request may need, and
    either of the calibrations described on `bytes_per_l2_chunk`; both fall back
    to the module defaults, so an empty dict still applies the default limit.
    Pass `None` to leave the request unrestricted.

    Raises `exceptions.RequestTooLarge` (HTTP 413) before any chunk is read.
    """
    if limits is None:
        return

    max_bytes = int(limits.get("MAX_BYTES", DEFAULT_MAX_BYTES))
    bytes_per_chunk = bytes_per_l2_chunk(cg.meta, limits)
    max_chunks = max(int(max_bytes // bytes_per_chunk), 1)

    n_chunks = level2_chunk_count(cg.meta, bounding_box)
    if n_chunks <= max_chunks:
        return

    x, y, z = _suggested_box(cg.meta, max_chunks)
    rx, ry, rz = [int(r) for r in cg.meta.resolution]
    gb = 1024.0**3

    scope = "the whole dataset" if bounding_box is None else "the requested bounds"
    raise cg_exceptions.RequestTooLarge(
        f"Subgraph request too large: {scope} spans {n_chunks} level 2 chunks, "
        f"which needs roughly {n_chunks * bytes_per_chunk / gb:.1f} GB of memory "
        f"to load (limit {max_bytes / gb:.1f} GB, {max_chunks} chunks). This "
        "query reads every edge stored in the chunks the box touches, so its "
        "cost scales with the volume of the box rather than with the size of "
        "the segment. Please split it into smaller boxes -- up to about "
        f"{x}x{y}x{z} voxels at {rx}x{ry}x{rz} nm resolution each -- and "
        "combine the results."
    )
