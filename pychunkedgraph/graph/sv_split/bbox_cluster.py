"""Dense-cluster bbox selection for SV-split seeds.

Public entry point: ``tight_bbox(cg, src_coords, sink_coords)`` — same shape
contract as ``edits._coords_bbox``. Picks the source / sink cluster pair
whose combined bbox is smallest, so outlier seeds don't stretch the
geodesic-cut bbox.

Outliers are not verified here. They surface downstream:

- The SV-split bbox computed by this module bounds only the geodesic
  voxel cut. The retry multicut in ``MulticutOperation._run_multicut``
  builds its own bbox from the full ``source_coords`` / ``sink_coords``
  (via ``get_bbox`` with ``split_bounding_offset``). The retry's local
  subgraph therefore spans the whole user-intent region, including
  voxels where the outlier seeds live.
- An outlier whose physical SV ended up on the wrong side of the cut
  manifests as a connectivity or root violation in the retry multicut:
  ``_filter_graph_connected_components`` raises
  ``PreconditionError: Not all sinks and sources are within the same
  (local) connected component`` (or a multi-root error from
  ``assert_same_root``).
- An outlier seed inside an SV piece that the cut didn't touch is
  still part of the post-split graph because the unsplit pieces keep
  their original IDs and edges. The retry mincut decides which side
  they belong to.

Net: the SV-split bbox shrinks the geodesic work; the retry multicut
remains the authoritative arbiter of whether the user's full seed
configuration is satisfiable.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from pychunkedgraph.graph.chunkedgraph import ChunkedGraph


def _coords_bbox(
    cg: "ChunkedGraph", src_coords: np.ndarray, sink_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Literal min/max of seeds + one-chunk margin, clipped to volume bounds."""
    coords = np.concatenate([src_coords, sink_coords], axis=0)
    margin = np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=int)
    vol_start = cg.meta.voxel_bounds[:, 0]
    vol_end = cg.meta.voxel_bounds[:, 1]
    bbs = np.clip(coords.min(axis=0) - margin, vol_start, vol_end)
    bbe = np.clip(coords.max(axis=0) + margin, vol_start, vol_end)
    return bbs, bbe


def _cluster_labels(coords_nm: np.ndarray, eps_nm: float) -> np.ndarray:
    """Group coords by connected components within ``eps_nm``.

    Returns ``(n,)`` int labels. Points within ``eps_nm`` of each other
    share a label.
    """
    n = len(coords_nm)
    if n == 0:
        return np.empty(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    tree = cKDTree(coords_nm)
    pairs = tree.query_pairs(r=eps_nm, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def _pick_dense_pair(
    src_labels: np.ndarray,
    sink_labels: np.ndarray,
    src_coords: np.ndarray,
    sink_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick (src cluster, sink cluster) with the most combined seeds, then
    the smallest combined bbox volume.

    Seed count carries user intent: the cluster with the most clicks is
    where the cut belongs. Volume only matters to break ties between
    equally-populated pairs (rare).
    """
    src_clusters = np.unique(src_labels)
    sink_clusters = np.unique(sink_labels)
    best_key = None
    best_pair = (int(src_clusters[0]), int(sink_clusters[0]))
    for s in src_clusters:
        s_mask = src_labels == s
        s_pts = src_coords[s_mask]
        for t in sink_clusters:
            t_mask = sink_labels == t
            t_pts = sink_coords[t_mask]
            size = int(s_mask.sum()) + int(t_mask.sum())
            all_pts = np.concatenate([s_pts, t_pts])
            extent = all_pts.max(axis=0) - all_pts.min(axis=0) + 1
            volume = float(np.prod(extent))
            key = (-size, volume)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (int(s), int(t))
    return src_labels == best_pair[0], sink_labels == best_pair[1]


def tight_bbox(
    cg: "ChunkedGraph",
    src_coords: np.ndarray,
    sink_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bbox of the densest source-cluster + sink-cluster pair plus one-chunk
    margin, clipped to ``cg.meta.voxel_bounds``.

    Matches ``edits._coords_bbox``'s shape contract. The cluster eps in
    nm is the average chunk extent in nm — points within one chunk in
    nm-space share a cluster. A side with a single seed becomes a
    single cluster (no shrink on that side); the other side still
    clusters and outliers there are excluded.
    """
    resolution = np.array(cg.meta.resolution, dtype=float)
    chunk_size = np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=float)
    eps_nm = float(np.mean(chunk_size * resolution))
    src_labels = _cluster_labels(src_coords * resolution, eps_nm)
    sink_labels = _cluster_labels(sink_coords * resolution, eps_nm)
    src_mask, sink_mask = _pick_dense_pair(
        src_labels, sink_labels, src_coords, sink_coords
    )
    return _coords_bbox(cg, src_coords[src_mask], sink_coords[sink_mask])
