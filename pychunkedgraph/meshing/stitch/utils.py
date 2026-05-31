# pylint: disable=invalid-name, missing-docstring, c-extension-no-member

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import zmesh

from pychunkedgraph.graph.chunks import utils as chunk_utils
from pychunkedgraph.graph.chunks import hierarchy as chunk_hierarchy
from pychunkedgraph.meshing import meshgen


@dataclass(frozen=True)
class MipInfo:
    """The watershed-CloudVolume values the geometry math needs, snapshotted in
    the parent so workers never touch ``meta.cv`` (tensorstore is fork-hostile).
    Built once via ``MipInfo.from_meta`` and passed to workers (picklable)."""

    mip: int
    resolution: Tuple[int, int, int]
    voxel_offset: Tuple[int, int, int]
    scales: List[dict]

    @classmethod
    def from_meta(cls, meta, mip):
        return cls(
            mip=int(mip),
            resolution=np.asarray(meta.cv.mip_resolution(mip)),
            voxel_offset=np.asarray(meta.cv.mip_voxel_offset(mip)),
            scales=meta.dataset_info["scales"],
        )


def mesh_block_shape_for_mip(meta, graphlayer, mip_info):
    """``meta``+``mip_info`` form of ``meshgen_utils.get_mesh_block_shape_for_mip``."""
    scales = mip_info.scales
    distortion = np.floor_divide(
        scales[mip_info.mip]["resolution"], scales[0]["resolution"]
    )
    graphlayer_chunksize = np.array(
        meta.graph_config.CHUNK_SIZE
    ) * meta.graph_config.FANOUT ** np.max([0, graphlayer - 2])
    return np.floor_divide(
        graphlayer_chunksize, distortion, dtype=int, casting="unsafe"
    )


def meshing_necessities(meta, chunk_id, mip_info):
    """``(layer, mesh_block_shape, chunk_offset)`` for ``chunk_id``, using
    parent-precomputed ``mip_info`` instead of ``meta.cv``."""
    layer = chunk_utils.get_chunk_layer(meta, chunk_id)
    cx, cy, cz = chunk_utils.get_chunk_coordinates(meta, chunk_id)
    mesh_block_shape = mesh_block_shape_for_mip(meta, layer, mip_info)
    chunk_offset = (
        (cx, cy, cz) * mesh_block_shape + mip_info.voxel_offset
    ) * mip_info.resolution
    return layer, mesh_block_shape, chunk_offset


def get_next_layer_draco_encoding_settings_pure(
    prev_layer_encoding_settings,
    mesh_block_shape,
    chunk_offset,
    segmentation_resolution,
):
    """``cg``-free form of ``meshgen.get_next_layer_draco_encoding_settings``."""
    old_draco_bin_size = prev_layer_encoding_settings["quantization_range"] // (
        2 ** prev_layer_encoding_settings["quantization_bits"] - 1
    )
    min_quantization_range = (
        max(mesh_block_shape * segmentation_resolution) + 2 * old_draco_bin_size
    )
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    (
        draco_quantization_bits,
        draco_quantization_range,
        draco_bin_size,
    ) = meshgen.calculate_quantization_bits_and_range(
        min_quantization_range, max_draco_bin_size
    )
    draco_quantization_origin = (
        chunk_offset
        - old_draco_bin_size
        - ((chunk_offset - old_draco_bin_size) % draco_bin_size)
    )
    return {
        "quantization_bits": draco_quantization_bits,
        "compression_level": 1,
        "quantization_range": draco_quantization_range,
        "quantization_origin": draco_quantization_origin,
        "create_metadata": True,
    }


def compute_layer_geometry(meta, node_id, layer, mip_info):
    """Per-layer-step ``(mesh_block_shape, chunk_offset, seg_resolution)`` for
    ``fragment_layer+1 .. layer``, from ``meta`` (chunk bit-math) +
    parent-precomputed ``mip_info`` (no ``meta.cv``)."""
    parent_chunk_ids = chunk_hierarchy.get_parent_chunk_ids(meta, node_id)
    fragment_layer = chunk_utils.get_chunk_layer(meta, node_id)
    if fragment_layer >= layer:
        raise ValueError(
            f"Node {node_id} somehow has greater or equal layer than target layer {layer}"
        )
    assert len(parent_chunk_ids) > layer - fragment_layer
    geometry = []
    for next_layer in range(fragment_layer + 1, layer + 1):
        next_layer_chunk_id = parent_chunk_ids[next_layer - fragment_layer]
        _, mesh_block_shape, chunk_offset = meshing_necessities(
            meta, next_layer_chunk_id, mip_info
        )
        geometry.append((mesh_block_shape, chunk_offset, mip_info.resolution))
    return geometry


def apply_draco_transform(mesh, layer_geometry, fragment_layer, layer):
    """``cg``-free form of ``transform_draco_fragment_and_return_encoding_options``.
    Mutates ``mesh["vertices"]`` per intermediate layer; returns final settings.
    ``mesh`` carries the decoded fragment's starting quantization as
    ``encoding_options_qr`` / ``encoding_options_qb``."""
    cur_encoding_settings = {
        "quantization_range": mesh["encoding_options_qr"],
        "quantization_bits": mesh["encoding_options_qb"],
    }
    for i, (mesh_block_shape, chunk_offset, segmentation_resolution) in enumerate(
        layer_geometry
    ):
        next_layer = fragment_layer + 1 + i
        next_encoding_settings = get_next_layer_draco_encoding_settings_pure(
            cur_encoding_settings,
            mesh_block_shape,
            chunk_offset,
            segmentation_resolution,
        )
        if next_layer < layer:
            meshgen.transform_draco_vertices(mesh, next_encoding_settings)
        cur_encoding_settings = next_encoding_settings
    return cur_encoding_settings


def get_draco_encoding_settings_for_chunk(meta, chunk_id, mip_info, high_padding):
    """``meta``+``mip_info`` form of ``meshgen.get_draco_encoding_settings_for_chunk``."""
    _, mesh_block_shape, chunk_offset = meshing_necessities(meta, chunk_id, mip_info)
    segmentation_resolution = mip_info.resolution
    min_quantization_range = max(
        (mesh_block_shape + high_padding) * segmentation_resolution
    )
    max_draco_bin_size = np.floor(min(segmentation_resolution) / np.sqrt(2))
    (
        draco_quantization_bits,
        draco_quantization_range,
        draco_bin_size,
    ) = meshgen.calculate_quantization_bits_and_range(
        min_quantization_range, max_draco_bin_size
    )
    draco_quantization_origin = chunk_offset - (chunk_offset % draco_bin_size)
    return {
        "quantization_bits": draco_quantization_bits,
        "compression_level": 1,
        "quantization_range": draco_quantization_range,
        "quantization_origin": draco_quantization_origin,
        "create_metadata": True,
    }


def compute_merge_boundary(meta, chunk_id, mip_info, high_padding):
    """Parent-chunk quantized boundary plane for ``merge_*_pure``; identical for
    every parent in the task. ``meta``+``mip_info`` form of meshgen lines 720-750."""
    chunk_coords = chunk_utils.get_chunk_coordinates(meta, chunk_id)
    coords_bottom_corner_child_chunk = chunk_coords * 2 + 1
    child_chunk_id = chunk_utils.get_chunk_id(
        meta,
        layer=chunk_utils.get_chunk_layer(meta, chunk_id) - 1,
        x=coords_bottom_corner_child_chunk[0],
        y=coords_bottom_corner_child_chunk[1],
        z=coords_bottom_corner_child_chunk[2],
    )
    _, _, child_chunk_offset = meshing_necessities(meta, child_chunk_id, mip_info)
    draco_encoding_settings_smaller_chunk = get_draco_encoding_settings_for_chunk(
        meta, child_chunk_id, mip_info, high_padding
    )
    draco_bin_size = draco_encoding_settings_smaller_chunk["quantization_range"] / (
        2 ** draco_encoding_settings_smaller_chunk["quantization_bits"] - 1
    )
    chunk_boundary_bin_index = np.floor(
        (
            child_chunk_offset
            - draco_encoding_settings_smaller_chunk["quantization_origin"]
        )
        / draco_bin_size
        + np.float32(0.5)
    )
    return (
        draco_encoding_settings_smaller_chunk["quantization_origin"]
        + chunk_boundary_bin_index * draco_bin_size
    )


def _remap_faces(faces, num_vertices, index_remaps):
    """Relabel ``faces`` (each entry an index into ``0..num_vertices-1``) to the
    deduped vertex indices.

    ``index_remaps`` is a list of ``(original_indices, new_indices)`` array pairs
    whose ``original_indices`` together cover every value in ``0..num_vertices-1``
    exactly once (each original vertex is either kept or folded into a boundary
    duplicate). Because that key range is contiguous, a lookup table indexed by
    original vertex id gives the same per-face relabeling a dictionary would —
    while avoiding a Python dictionary with one entry per vertex (which grows to
    the full vertex count, millions on a large parent, and dominates merge time
    and memory). The table is scattered from the pairs, then gathered by faces.
    """
    lookup_table = np.empty(num_vertices, dtype=np.uint32)
    for original_indices, new_indices in index_remaps:
        lookup_table[original_indices] = new_indices
    return lookup_table[faces]


def merge_draco_meshes_across_boundaries_pure(
    fragments, quantized_chunk_boundary, return_zmesh_object=False
):
    """``cg``-free merge: concatenate fragment vertices/faces and dedup vertices
    on the (precomputed) quantized chunk boundary. meshgen lines 711-717 + 752-794."""
    vertexct = np.zeros(len(fragments) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x["mesh"]["num_vertices"] for x in fragments])
    vertices = np.concatenate([x["mesh"]["vertices"] for x in fragments])
    faces = np.concatenate(
        [mesh["mesh"]["faces"] + vertexct[i] for i, mesh in enumerate(fragments)]
    )
    del fragments

    if vertexct[-1] > 0:
        # Carry each vertex's original index out of band (uint32) rather than
        # hstacking it onto the float32 vertices — that hstack with an int64
        # column promotes everything to float64, doubling the vertex buffer and
        # running the dedup np.unique on float64. Splitting by the boolean mask
        # keeps vertices float32 throughout (the values are unchanged; float32
        # is exactly representable, so the dedup result is identical).
        are_chunk_aligned = (vertices == quantized_chunk_boundary).any(axis=1)
        original_index = np.arange(vertexct[-1], dtype=np.uint32)
        chunk_aligned = vertices[are_chunk_aligned]
        not_chunk_aligned = vertices[~are_chunk_aligned]
        chunk_aligned_index = original_index[are_chunk_aligned]
        not_chunk_aligned_index = original_index[~are_chunk_aligned]
        del vertices
        del are_chunk_aligned
        index_remaps = []
        if len(not_chunk_aligned) > 0:
            index_remaps.append(
                (
                    not_chunk_aligned_index,
                    np.arange(len(not_chunk_aligned), dtype=np.uint32),
                )
            )
        if len(chunk_aligned) > 0:
            unique_chunk_aligned, inverse_to_chunk_aligned = np.unique(
                chunk_aligned, return_inverse=True, axis=0
            )
            index_remaps.append(
                (
                    chunk_aligned_index,
                    np.uint32(len(not_chunk_aligned))
                    + inverse_to_chunk_aligned.reshape(-1).astype(np.uint32),
                )
            )
            vertices = np.concatenate((not_chunk_aligned, unique_chunk_aligned))
        else:
            vertices = not_chunk_aligned
        faces = _remap_faces(faces, vertexct[-1], index_remaps)

    if return_zmesh_object:
        return zmesh.Mesh(vertices, faces.reshape(-1, 3), None)

    return {
        "num_vertices": np.uint32(len(vertices)),
        "vertices": vertices.reshape(-1),
        "faces": faces,
    }


MIN_BATCH = 32


def derive_batch_size(n_parents, n_processes, batches_per_worker=64):
    """Split into ~``n_processes * batches_per_worker`` batches so work-stealing
    is fine enough to balance the heavy-tailed per-parent cost. ``MIN_BATCH``
    floors the size so each batch still coalesces enough byte ranges to amortize
    its minishard-index reads."""
    target_batches = max(1, n_processes * batches_per_worker)
    return max(MIN_BATCH, math.ceil(n_parents / target_batches))


def sample_debug(multi_child_nodes, max_parents):
    """Down-sample ``{parent_id: [child_ids]}`` for a debug preview: stride
    evenly across the smaller half (proxy = immediate-child count), spanning
    just-above-medium down to tiny. Skips the giant stragglers whose encode
    dominates wall time so the preview exercises every stage cheaply. Returns
    the dict unchanged when it already has ``<= max_parents`` entries."""
    if len(multi_child_nodes) <= max_parents:
        return multi_child_nodes
    # ascending by size; the lower half is medium..tiny (drop the big upper half).
    ranked = sorted(multi_child_nodes, key=lambda p: (len(multi_child_nodes[p]), p))
    lower = ranked[: max(max_parents, len(ranked) // 2)]
    step = len(lower) / max_parents
    sampled = [lower[int(i * step)] for i in range(max_parents)]
    return {p: multi_child_nodes[p] for p in sampled}
