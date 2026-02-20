# pylint: disable=invalid-name, missing-docstring, too-many-lines, wrong-import-order, import-outside-toplevel, no-member, c-extension-no-member

from typing import Sequence
import collections
import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
import DracoPy
import fastremap

from pychunkedgraph.graph.chunkedgraph import ChunkedGraph  # noqa
from pychunkedgraph.graph import attributes  # noqa
from pychunkedgraph.meshing import meshgen_utils
from pychunkedgraph.meshing.meshgen_utils import (
    UTC,
    PRINT_FOR_DEBUGGING,
    WRITING_TO_CLOUD,
    remap_seg_using_unsafe_dict,
    calculate_stop_layer,
    get_multi_child_nodes,
    transform_draco_fragment_and_return_encoding_options,
    merge_draco_meshes_across_boundaries,
)
from pychunkedgraph.meshing.meshgen_initial import chunk_initial_mesh_task
from pychunkedgraph.meshing.manifest.cache import ManifestCache


def get_remapped_seg_for_lvl2_nodes(
    cg,
    chunk_id: np.uint64,
    lvl2_nodes: Sequence[np.uint64],
    mip: int = 2,
    overlap_vx: int = 1,
    time_stamp=None,
    n_threads: int = 1,
):
    """Downloads + remaps ws segmentation + resolve unclear cases,
    filter out all but specified lvl2_nodes

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
    :param overlap_vx: int
    :param time_stamp:
    :return: remapped segmentation
    """
    seg = meshgen_utils.get_ws_seg_for_chunk(cg, chunk_id, mip, overlap_vx)
    sv_of_lvl2_nodes = cg.get_children(lvl2_nodes)

    # Check which of the lvl2_nodes meet the chunk boundary
    node_ids_on_the_border = []
    remapping = {}
    for node, sv_list in sv_of_lvl2_nodes.items():
        node_on_the_border = False
        for sv_id in sv_list:
            remapping[sv_id] = node
            # If a node_id is on the chunk_boundary, we must check
            # the overlap region to see if the meshes' end will be open or closed
            if (not node_on_the_border) and (
                np.isin(sv_id, seg[-2, :, :])
                or np.isin(sv_id, seg[:, -2, :])
                or np.isin(sv_id, seg[:, :, -2])
            ):
                node_on_the_border = True
                node_ids_on_the_border.append(node)

    node_ids_on_the_border = np.array(node_ids_on_the_border)
    if len(node_ids_on_the_border) > 0:
        overlap_region = np.concatenate(
            (seg[:, :, -1], seg[:, -1, :], seg[-1, :, :]), axis=None
        )
        overlap_sv_ids = np.unique(overlap_region)
        if overlap_sv_ids[0] == 0:
            overlap_sv_ids = overlap_sv_ids[1:]
        # Get the remappings for the supervoxels in the overlap region
        sv_remapping, unsafe_dict = get_lx_overlapping_remappings_for_nodes_and_svs(
            cg, chunk_id, node_ids_on_the_border, overlap_sv_ids, time_stamp, n_threads
        )
        sv_remapping.update(remapping)
        fastremap.mask_except(seg, list(sv_remapping.keys()), in_place=True)
        fastremap.remap(seg, sv_remapping, preserve_missing_labels=True, in_place=True)
        # For some supervoxel, they could map to multiple l2 nodes in the chunk,
        # so we must perform a connected component analysis
        # to see which l2 node they are adjacent to
        return remap_seg_using_unsafe_dict(seg, unsafe_dict)
    else:
        # If no nodes in our subset meet the chunk boundary
        # we can simply retrieve the sv of the nodes in the subset
        fastremap.mask_except(seg, list(remapping.keys()), in_place=True)
        fastremap.remap(seg, remapping, preserve_missing_labels=True, in_place=True)

    return seg


def get_root_remapping_for_nodes_and_svs(
    cg, chunk_id, node_ids, sv_ids, stop_layer, time_stamp, n_threads=1
):
    """Retrieves root to node id mapping for specified node ids and supervoxel ids

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param node_ids: [np.uint64]
    :param stop_layer: int
    :param time_stamp: datetime object
    :return: multiples
    """

    def _get_root_ids(args):
        start_id, end_id = args

        root_ids[start_id:end_id] = cg.get_roots(
            combined_ids[start_id:end_id],
            stop_layer=stop_layer,
            time_stamp=time_stamp,
            fail_to_zero=True,
        )

    rr = cg.range_read_chunk(
        chunk_id=chunk_id, properties=attributes.Hierarchy.Child, time_stamp=time_stamp
    )
    chunk_sv_ids = np.unique(np.concatenate([id[0].value for id in rr.values()]))
    chunk_l2_ids = np.unique(cg.get_parents(chunk_sv_ids, time_stamp=time_stamp))
    combined_ids = np.concatenate((node_ids, sv_ids, chunk_l2_ids))

    root_ids = np.zeros(len(combined_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(combined_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(combined_ids), n_jobs + 1).astype(int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            list(executor.map(_get_root_ids, multi_args))

    sv_ids_index = len(node_ids)
    chunk_ids_index = len(node_ids) + len(sv_ids)

    return (
        root_ids[0:sv_ids_index],
        root_ids[sv_ids_index:chunk_ids_index],
        root_ids[chunk_ids_index:],
    )


def get_lx_overlapping_remappings_for_nodes_and_svs(
    cg,
    chunk_id: np.uint64,
    node_ids: Sequence[np.uint64],
    sv_ids: Sequence[np.uint64],
    time_stamp=None,
    n_threads: int = 1,
):
    """Retrieves sv id to layer mapping for chunk with overlap in positive
        direction (one chunk)

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param node_ids: list of np.uint64
    :param sv_ids: list of np.uint64
    :param time_stamp: datetime object
    :param n_threads: int
    :return: multiples
    """
    if time_stamp is None:
        time_stamp = datetime.datetime.now(datetime.timezone.utc)
    if time_stamp.tzinfo is None:
        time_stamp = UTC.localize(time_stamp)

    stop_layer, _ = calculate_stop_layer(cg, chunk_id)
    print(f"Stop layer: {stop_layer}")

    # Find the parent in the lowest common chunk for each node id and sv id. These parent
    # ids are referred to as root ids even though they are not necessarily the
    # root id.
    node_root_ids, sv_root_ids, chunks_root_ids = get_root_remapping_for_nodes_and_svs(
        cg, chunk_id, node_ids, sv_ids, stop_layer, time_stamp, n_threads
    )

    u_root_ids, u_idx, c_root_ids = np.unique(
        chunks_root_ids, return_counts=True, return_index=True
    )

    # All l2 ids that share no root id with any other l2 id in the chunk are "safe", meaning
    # that we can easily obtain the complete remapping (including
    # overlap) for these. All other ones have to be resolved using the
    # segmentation.

    root_sorted_idx = np.argsort(u_root_ids)
    node_sorted_index = np.searchsorted(u_root_ids[root_sorted_idx], node_root_ids)
    node_root_counts = c_root_ids[root_sorted_idx][node_sorted_index]
    unsafe_root_ids = node_root_ids[np.where(node_root_counts > 1)]
    safe_node_ids = node_ids[~np.isin(node_root_ids, unsafe_root_ids)]

    node_to_root_dict = dict(zip(node_ids, node_root_ids))

    # Future sv id -> lx mapping
    sv_ids_to_remap = []
    node_ids_flat = []

    # Do safe ones first
    for node_id in safe_node_ids:
        root_id = node_to_root_dict[node_id]
        sv_ids_to_add = sv_ids[np.where(sv_root_ids == root_id)]
        if len(sv_ids_to_add) > 0:
            sv_ids_to_remap.extend(sv_ids_to_add)
            node_ids_flat.extend([node_id] * len(sv_ids_to_add))

    # For the unsafe roots, we will map the out of chunk svs to the root id and store the
    # hierarchical information in a dictionary
    unsafe_dict = collections.defaultdict(list)
    for root_id in unsafe_root_ids:
        sv_ids_to_add = sv_ids[np.where(sv_root_ids == root_id)]
        if len(sv_ids_to_add) > 0:
            relevant_node_ids = node_ids[np.where(node_root_ids == root_id)]
            if len(relevant_node_ids) > 0:
                unsafe_dict[root_id].extend(relevant_node_ids)
                sv_ids_to_remap.extend(sv_ids_to_add)
                node_ids_flat.extend([root_id] * len(sv_ids_to_add))

    # Combine the lists for a (chunk-) global remapping
    sv_remapping = dict(zip(sv_ids_to_remap, node_ids_flat))

    return sv_remapping, unsafe_dict


def _get_timestamp_from_node_ids(cg, node_ids):
    timestamps = cg.get_node_timestamps(node_ids, return_numpy=False)
    return max(timestamps) + datetime.timedelta(milliseconds=1)


def remeshing(
    cg,
    l2_node_ids: Sequence[np.uint64],
    cv_sharded_mesh_dir: str,
    cv_unsharded_mesh_path: str,
    stop_layer: int = None,
    mip: int = 2,
    max_err: int = 40,
    time_stamp: datetime.datetime or None = None,
):
    """Given a chunkedgraph, a list of level 2 nodes,
    perform remeshing and stitching up the node hierarchy (or up to the stop_layer)

    :param cg: chunkedgraph instance
    :param l2_node_ids: list of uint64
    :param stop_layer: int
    :param cv_path: str
    :param cv_mesh_dir: str
    :param mip: int
    :param max_err: int
    :return:
    """
    l2_chunk_dict = collections.defaultdict(set)
    # Find the chunk_ids of the l2_node_ids

    def add_nodes_to_l2_chunk_dict(ids):
        for node_id in ids:
            chunk_id = cg.get_chunk_id(node_id)
            l2_chunk_dict[chunk_id].add(node_id)

    add_nodes_to_l2_chunk_dict(l2_node_ids)
    for chunk_id, node_ids in l2_chunk_dict.items():
        if PRINT_FOR_DEBUGGING:
            print("remeshing", chunk_id, node_ids)
        try:
            l2_time_stamp = _get_timestamp_from_node_ids(cg, node_ids)
        except ValueError:
            # ignore bad/invalid messages
            return
        # Remesh the l2_node_ids
        chunk_initial_mesh_task(
            None,
            chunk_id,
            mip=mip,
            node_id_subset=node_ids,
            cg=cg,
            cv_unsharded_mesh_path=cv_unsharded_mesh_path,
            max_err=max_err,
            sharded=False,
            time_stamp=l2_time_stamp,
        )
    chunk_dicts = []
    max_layer = stop_layer or cg._n_layers
    for layer in range(3, max_layer + 1):
        chunk_dicts.append(collections.defaultdict(set))
    cur_chunk_dict = l2_chunk_dict
    # Find the parents of each l2_node_id up to the stop_layer,
    # as well as their associated chunk_ids
    for layer in range(3, max_layer + 1):
        for _, node_ids in cur_chunk_dict.items():
            parent_nodes = cg.get_parents(node_ids, time_stamp=time_stamp)
            for parent_node in parent_nodes:
                chunk_layer = cg.get_chunk_layer(parent_node)
                index_in_dict_array = chunk_layer - 3
                if index_in_dict_array < len(chunk_dicts):
                    chunk_id = cg.get_chunk_id(parent_node)
                    chunk_dicts[index_in_dict_array][chunk_id].add(parent_node)
        cur_chunk_dict = chunk_dicts[layer - 3]
    for chunk_dict in chunk_dicts:
        for chunk_id, node_ids in chunk_dict.items():
            if PRINT_FOR_DEBUGGING:
                print("remeshing", chunk_id, node_ids)
            # Stitch the meshes of the parents we found in the previous loop
            chunk_stitch_remeshing_task(
                None,
                chunk_id,
                mip=mip,
                fragment_batch_size=40,
                node_id_subset=node_ids,
                cg=cg,
                cv_sharded_mesh_dir=cv_sharded_mesh_dir,
                cv_unsharded_mesh_path=cv_unsharded_mesh_path,
            )


def chunk_stitch_remeshing_task(
    cg_name,
    chunk_id,
    cv_sharded_mesh_dir,
    cv_unsharded_mesh_path,
    mip=2,
    lod=0,
    fragment_batch_size=None,
    node_id_subset=None,
    cg=None,
    high_padding=1,
):
    """
    For each node with more than one child, create a new fragment by
    merging the mesh fragments of the children.
    """
    if cg is None:
        cg = ChunkedGraph(graph_id=cg_name)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    layer = cg.get_chunk_layer(chunk_id)
    result = []

    assert layer > 2

    print(
        "Retrieving children for chunk %s -- (%s, %s, %s, %s)"
        % (chunk_id, layer, cx, cy, cz)
    )

    multi_child_nodes, _ = get_multi_child_nodes(cg, chunk_id, node_id_subset, False)
    print(f"{len(multi_child_nodes)} nodes with more than one child")
    result.append((chunk_id, len(multi_child_nodes)))
    if not multi_child_nodes:
        print("Nothing to do", cx, cy, cz)
        return ", ".join(str(x) for x in result)

    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=cv_sharded_mesh_dir,
        info=meshgen_utils.get_json_info(cg),
    )

    fragment_to_fetch = [
        fragment
        for child_fragments in multi_child_nodes.values()
        for fragment in child_fragments
    ]

    # Fetch all fragments upfront for parallel processing
    fragment_map = cv.mesh.get_meshes_on_bypass(fragment_to_fetch, allow_missing=True)

    # Process each node's fragments in parallel
    def _process_stitch_node(item):
        new_fragment_id, fragment_ids_to_fetch = item

        old_fragments = []
        missing_fragments = False
        for fragment_id in fragment_ids_to_fetch:
            if fragment_id in fragment_map:
                old_frag = fragment_map[fragment_id]
                new_old_frag = {
                    "num_vertices": len(old_frag.vertices),
                    "vertices": old_frag.vertices,
                    "faces": old_frag.faces.reshape(-1),
                    "encoding_options": old_frag.encoding_options,
                    "encoding_type": "draco",
                }
                wrapper_object = {
                    "mesh": new_old_frag,
                    "node_id": np.uint64(old_frag.segid),
                }
                old_fragments.append(wrapper_object)
            elif cg.get_chunk_layer(np.uint64(fragment_id)) > 2:
                missing_fragments = True
                return (
                    new_fragment_id,
                    None,
                    None,
                    f"{fragment_id} missing for {new_fragment_id}",
                )

        if len(old_fragments) == 0 or missing_fragments:
            return (new_fragment_id, None, None, f"No meshes for {new_fragment_id}")

        draco_encoding_options = None
        for old_fragment in old_fragments:
            if draco_encoding_options is None:
                draco_encoding_options = (
                    transform_draco_fragment_and_return_encoding_options(
                        cg, old_fragment, layer, mip, chunk_id
                    )
                )
            else:
                transform_draco_fragment_and_return_encoding_options(
                    cg, old_fragment, layer, mip, chunk_id
                )

        new_fragment = merge_draco_meshes_across_boundaries(
            cg, old_fragments, chunk_id, mip, high_padding
        )

        try:
            new_fragment_b = DracoPy.encode_mesh_to_buffer(
                new_fragment["vertices"],
                new_fragment["faces"],
                **draco_encoding_options,
            )
        except:
            return (
                new_fragment_id,
                None,
                None,
                f'Bad mesh created for {new_fragment_id}: {len(new_fragment["vertices"])} '
                f'vertices, {len(new_fragment["faces"])} faces',
            )

        fragment_name = None
        if WRITING_TO_CLOUD:
            fragment_name = meshgen_utils.get_chunk_bbox_str(cg, new_fragment_id)
            fragment_name = f"{new_fragment_id}:0:{fragment_name}"
            thread_cf = CloudFiles(cv_unsharded_mesh_path)
            thread_cf.put(
                fragment_name,
                new_fragment_b,
                content_type="application/octet-stream",
                compress=False,
                cache_control="public",
            )

        return (new_fragment_id, fragment_name, new_fragment_b, None)

    fragments_d = {}
    with ThreadPoolExecutor() as executor:
        for r in executor.map(_process_stitch_node, multi_child_nodes.items()):
            new_fragment_id, fragment_name, _, error = r
            if error is not None:
                result.append(error)
            elif fragment_name is not None:
                fragments_d[new_fragment_id] = fragment_name

    manifest_cache = ManifestCache(cg.graph_id, initial=False)
    manifest_cache.set_fragments(fragments_d)

    if PRINT_FOR_DEBUGGING:
        print(", ".join(str(x) for x in result))
    return ", ".join(str(x) for x in result)
