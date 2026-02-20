# pylint: disable=invalid-name, missing-docstring, too-many-lines, wrong-import-order, import-outside-toplevel, no-member, c-extension-no-member

import os
import collections
import datetime
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
import DracoPy
import zmesh
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
    get_meshing_necessities_from_graph,
    get_draco_encoding_settings_for_chunk,
    black_out_dust_from_segmentation,
    get_multi_child_nodes,
    decode_draco_mesh_buffer,
    transform_draco_fragment_and_return_encoding_options,
    merge_draco_meshes_across_boundaries,
)


def get_remapped_segmentation(
    cg, chunk_id, mip=2, overlap_vx=1, time_stamp=None, n_threads=1
):
    """Downloads + remaps ws segmentation + resolve unclear cases

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param mip: int
    :param overlap_vx: int
    :param time_stamp:
    :return: remapped segmentation
    """
    assert mip >= cg.meta.cv.mip

    sv_remapping, unsafe_dict = get_lx_overlapping_remappings(
        cg, chunk_id, time_stamp=time_stamp, n_threads=n_threads
    )

    ws_seg = meshgen_utils.get_ws_seg_for_chunk(cg, chunk_id, mip, overlap_vx)
    seg = fastremap.mask_except(ws_seg, list(sv_remapping.keys()), in_place=False)
    fastremap.remap(seg, sv_remapping, preserve_missing_labels=True, in_place=True)

    return remap_seg_using_unsafe_dict(seg, unsafe_dict)


@lru_cache(maxsize=None)
def get_higher_to_lower_remapping(cg, chunk_id, time_stamp):
    """Retrieves lx node id to sv id mappping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param time_stamp: datetime object
    :return: dictionary
    """

    def _lower_remaps(ks):
        return np.concatenate([lower_remaps[k] for k in ks])

    assert cg.get_chunk_layer(chunk_id) >= 2
    assert cg.get_chunk_layer(chunk_id) <= cg.meta.layer_count

    print(f"\n{chunk_id} ----------------\n")

    lower_remaps = {}
    if cg.get_chunk_layer(chunk_id) > 2:
        for lower_chunk_id in cg.get_chunk_child_ids(chunk_id):
            # TODO speedup
            lower_remaps.update(
                get_higher_to_lower_remapping(cg, lower_chunk_id, time_stamp=time_stamp)
            )

    rr_chunk = cg.range_read_chunk(
        chunk_id=chunk_id, properties=attributes.Hierarchy.Child, time_stamp=time_stamp
    )

    # This for-loop ensures that only the latest lx_ids are considered
    # The order by id guarantees the time order (only true for same neurons
    # but that is the case here).
    lx_remapping = {}
    all_lower_ids = set()
    for k in sorted(rr_chunk.keys(), reverse=True):
        this_child_ids = rr_chunk[k][0].value
        if this_child_ids[0] in all_lower_ids:
            continue

        all_lower_ids = all_lower_ids.union(set(list(this_child_ids)))

        if cg.get_chunk_layer(chunk_id) > 2:
            try:
                lx_remapping[k] = _lower_remaps(this_child_ids)
            except KeyError:
                # KeyErrors indicate that this id is deprecated given the
                # time_stamp
                continue
        else:
            lx_remapping[k] = this_child_ids

    return lx_remapping


@lru_cache(maxsize=None)
def get_root_lx_remapping(cg, chunk_id, stop_layer, time_stamp, n_threads=1):
    """Retrieves root to l2 node id mapping

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param stop_layer: int
    :param time_stamp: datetime object
    :return: multiples
    """

    def _get_root_ids(args):
        start_id, end_id = args
        root_ids[start_id:end_id] = cg.get_roots(
            lx_ids[start_id:end_id],
            stop_layer=stop_layer,
            fail_to_zero=True,
        )

    lx_id_remap = get_higher_to_lower_remapping(cg, chunk_id, time_stamp=time_stamp)

    lx_ids = np.array(list(lx_id_remap.keys()))

    root_ids = np.zeros(len(lx_ids), dtype=np.uint64)
    n_jobs = np.min([n_threads, len(lx_ids)])
    multi_args = []
    start_ids = np.linspace(0, len(lx_ids), n_jobs + 1).astype(int)
    for i_block in range(n_jobs):
        multi_args.append([start_ids[i_block], start_ids[i_block + 1]])

    if n_jobs > 0:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            list(executor.map(_get_root_ids, multi_args))

    return lx_ids, np.array(root_ids), lx_id_remap


# @lru_cache(maxsize=None)
def get_lx_overlapping_remappings(cg, chunk_id, time_stamp=None, n_threads=1):
    """Retrieves sv id to layer mapping for chunk with overlap in positive
        direction (one chunk)

    :param cg: chunkedgraph object
    :param chunk_id: np.uint64
    :param time_stamp: datetime object
    :return: multiples
    """
    if time_stamp is None:
        time_stamp = datetime.datetime.now(datetime.timezone.utc)
    if time_stamp.tzinfo is None:
        time_stamp = UTC.localize(time_stamp)

    stop_layer, neigh_chunk_ids = calculate_stop_layer(cg, chunk_id)
    print(f"Stop layer: {stop_layer}")

    # Find the parent in the lowest common chunk for each l2 id. These parent
    # ids are referred to as root ids even though they are not necessarily the
    # root id.
    neigh_lx_ids = []
    neigh_lx_id_remap = {}
    neigh_root_ids = []

    safe_lx_ids = []
    unsafe_lx_ids = []
    unsafe_root_ids = []

    # Parallelize the main bottleneck: fetching root mappings for neighbor chunks
    with ThreadPoolExecutor() as executor:
        future_to_chunk = {
            executor.submit(
                get_root_lx_remapping,
                cg,
                nid,
                stop_layer,
                time_stamp=time_stamp,
                n_threads=n_threads,
            ): nid
            for nid in neigh_chunk_ids
        }
        results = {}
        for future in as_completed(future_to_chunk):
            nid = future_to_chunk[future]
            results[nid] = future.result()

    for neigh_chunk_id in neigh_chunk_ids:
        print(f"Neigh: {neigh_chunk_id} --------------")

        lx_ids, root_ids, lx_id_remap = results[neigh_chunk_id]
        neigh_lx_ids.extend(lx_ids)
        neigh_lx_id_remap.update(lx_id_remap)
        neigh_root_ids.extend(root_ids)

        if neigh_chunk_id == chunk_id:
            # The first neigh_chunk_id is the one we are interested in. All lx
            # ids that share no root id with any other lx id are "safe", meaning
            # that we can easily obtain the complete remapping (including
            # overlap) for these. All other ones have to be resolved using the
            # segmentation.
            _, u_idx, c_root_ids = np.unique(
                neigh_root_ids, return_counts=True, return_index=True
            )

            safe_lx_ids = lx_ids[u_idx[c_root_ids == 1]]
            unsafe_lx_ids = lx_ids[~np.isin(lx_ids, safe_lx_ids)]
            unsafe_root_ids = np.unique(root_ids[u_idx[c_root_ids != 1]])

    lx_root_dict = dict(zip(neigh_lx_ids, neigh_root_ids))
    root_lx_dict = collections.defaultdict(list)

    # Future sv id -> lx mapping
    sv_ids = []
    lx_ids_flat = []

    for i_root_id in range(len(neigh_root_ids)):
        root_lx_dict[neigh_root_ids[i_root_id]].append(neigh_lx_ids[i_root_id])

    # Do safe ones first
    for lx_id in safe_lx_ids:
        root_id = lx_root_dict[lx_id]
        for neigh_lx_id in root_lx_dict[root_id]:
            lx_sv_ids = neigh_lx_id_remap[neigh_lx_id]
            sv_ids.extend(lx_sv_ids)
            lx_ids_flat.extend([lx_id] * len(neigh_lx_id_remap[neigh_lx_id]))

    # For the unsafe ones we can only do the in chunk svs
    # But we will map the out of chunk svs to the root id and store the
    # hierarchical information in a dictionary
    for lx_id in unsafe_lx_ids:
        sv_ids.extend(neigh_lx_id_remap[lx_id])
        lx_ids_flat.extend([lx_id] * len(neigh_lx_id_remap[lx_id]))

    unsafe_dict = collections.defaultdict(list)
    for root_id in unsafe_root_ids:
        if np.sum(~np.isin(root_lx_dict[root_id], unsafe_lx_ids)) == 0:
            continue

        for neigh_lx_id in root_lx_dict[root_id]:
            unsafe_dict[root_id].append(neigh_lx_id)

            if neigh_lx_id in unsafe_lx_ids:
                continue

            sv_ids.extend(neigh_lx_id_remap[neigh_lx_id])
            lx_ids_flat.extend([root_id] * len(neigh_lx_id_remap[neigh_lx_id]))

    # Combine the lists for a (chunk-) global remapping
    sv_remapping = dict(zip(sv_ids, lx_ids_flat))

    return sv_remapping, unsafe_dict


def chunk_initial_mesh_task(
    cg_name,
    chunk_id,
    cv_unsharded_mesh_path,
    mip=2,
    max_err=40,
    lod=0,
    encoding="draco",
    time_stamp=None,
    dust_threshold=None,
    return_frag_count=False,
    node_id_subset=None,
    cg=None,
    sharded=False,
    cache=True,
):
    if cg is None:
        cg = ChunkedGraph(graph_id=cg_name)
    result = []
    cache_string = "public" if cache else "no-cache"

    layer, _, chunk_offset = get_meshing_necessities_from_graph(cg, chunk_id, mip)
    cx, cy, cz = cg.get_chunk_coordinates(chunk_id)
    high_padding = 1
    assert layer == 2
    assert mip >= cg.meta.cv.mip

    if sharded:
        cv = CloudVolume(
            f"graphene://https://localhost/segmentation/table/dummy",
            info=meshgen_utils.get_json_info(cg),
        )
        sharding_info = cv.mesh.meta.info["sharding"]["2"]
        sharding_spec = ShardingSpecification.from_dict(sharding_info)
        merged_meshes = {}
        mesh_dst = os.path.join(
            cv.cloudpath, cv.mesh.meta.mesh_path, "initial", str(layer)
        )
    else:
        mesh_dst = cv_unsharded_mesh_path

    result.append((chunk_id, layer, cx, cy, cz))
    print(
        "Retrieving remap table for chunk %s -- (%s, %s, %s, %s)"
        % (chunk_id, layer, cx, cy, cz)
    )
    mesher = zmesh.Mesher(cg.meta.cv.mip_resolution(mip))
    draco_encoding_settings = get_draco_encoding_settings_for_chunk(
        cg, chunk_id, mip, high_padding
    )
    if node_id_subset is None:
        seg = get_remapped_segmentation(
            cg, chunk_id, mip, overlap_vx=high_padding, time_stamp=time_stamp
        )
    else:
        # Import here to avoid circular import at module level
        from pychunkedgraph.meshing.meshgen_remesh import (
            get_remapped_seg_for_lvl2_nodes,
        )

        seg = get_remapped_seg_for_lvl2_nodes(
            cg,
            chunk_id,
            node_id_subset,
            mip=mip,
            overlap_vx=high_padding,
            time_stamp=time_stamp,
        )
    if dust_threshold:
        black_out_dust_from_segmentation(seg, dust_threshold)
    if return_frag_count:
        return np.unique(seg).shape[0]
    mesher.mesh(seg)
    del seg

    if PRINT_FOR_DEBUGGING:
        print("cv path", mesh_dst)
        print("num ids", len(mesher.ids()))
    result.append(len(mesher.ids()))

    # Extract all meshes sequentially (zmesh Mesher is not thread-safe)
    meshes = []
    for obj_id in mesher.ids():
        mesh = mesher.get(obj_id, reduction_factor=100, max_error=max_err)
        mesher.erase(obj_id)
        mesh.vertices[:] += chunk_offset
        meshes.append((obj_id, mesh))
    del mesher

    # Encode + upload in parallel
    def _encode_and_upload(args):
        obj_id, mesh = args
        if encoding == "draco":
            try:
                file_contents = DracoPy.encode_mesh_to_buffer(
                    mesh.vertices.flatten("C"),
                    mesh.faces.flatten("C"),
                    **draco_encoding_settings,
                )
            except:
                return (
                    "error",
                    f"{obj_id} failed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces",
                )
            compress = False
        else:
            file_contents = mesh.to_precomputed()
            compress = True
        if WRITING_TO_CLOUD:
            if sharded:
                return ("shard", int(obj_id), file_contents)
            else:
                thread_cf = CloudFiles(mesh_dst)
                thread_cf.put(
                    path=f"{meshgen_utils.get_mesh_name(cg, obj_id)}",
                    content=file_contents,
                    compress=compress,
                    cache_control=cache_string,
                )
        return None

    with ThreadPoolExecutor() as executor:
        for r in executor.map(_encode_and_upload, meshes):
            if r is None:
                continue
            if r[0] == "error":
                result.append(r[1])
            elif r[0] == "shard":
                merged_meshes[r[1]] = r[2]

    if sharded and WRITING_TO_CLOUD:
        shard_binary = sharding_spec.synthesize_shard(merged_meshes)
        shard_filename = cv.mesh.readers[layer].get_filename(chunk_id)
        cf = CloudFiles(mesh_dst)
        cf.put(
            shard_filename,
            shard_binary,
            content_type="application/octet-stream",
            compress=False,
            cache_control=cache_string,
        )
    if PRINT_FOR_DEBUGGING:
        print(", ".join(str(x) for x in result))
    return result


# ---------------------------------------------------------------------------
# Process-pool worker for chunk_initial_sharded_stitching_task
# ---------------------------------------------------------------------------

_worker_cg = None


def _init_cg_worker(graph_id):
    global _worker_cg
    _worker_cg = ChunkedGraph(graph_id=graph_id)


def _process_sharded_fragment(
    new_fragment_id, fragment_ids, frag_mesh_data, layer, mip, chunk_id, high_padding
):
    cg = _worker_cg
    old_fragments = []
    for frag_id in fragment_ids:
        frag_data = frag_mesh_data.get(int(frag_id))
        if frag_data is not None:
            try:
                old_fragments.append(
                    {
                        "mesh": decode_draco_mesh_buffer(frag_data),
                        "node_id": np.uint64(frag_id),
                    }
                )
            except (KeyError, ValueError):
                pass

    if not old_fragments:
        return (new_fragment_id, None, 0, None)

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

    vertex_count = len(new_fragment["vertices"])

    try:
        new_fragment_b = DracoPy.encode_mesh_to_buffer(
            new_fragment["vertices"],
            new_fragment["faces"],
            **draco_encoding_options,
        )
        return (new_fragment_id, new_fragment_b, vertex_count, None)
    except Exception:
        return (
            new_fragment_id,
            None,
            vertex_count,
            f"failed to merge {new_fragment_id}",
        )


def chunk_initial_sharded_stitching_task(
    cg_name, chunk_id, mip, cg=None, high_padding=1, cache=True
):
    start_existence_check_time = time.time()
    if cg is None:
        cg = ChunkedGraph(graph_id=cg_name)

    cache_string = "public" if cache else "no-cache"

    layer = cg.get_chunk_layer(chunk_id)
    multi_child_nodes, multi_child_descendants = get_multi_child_nodes(cg, chunk_id)

    chunk_to_id_dict = collections.defaultdict(list)
    for child_node in multi_child_descendants:
        cur_chunk_id = int(cg.get_chunk_id(child_node))
        chunk_to_id_dict[cur_chunk_id].append(child_node)

    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        info=meshgen_utils.get_json_info(cg),
    )
    shard_filenames = []
    shard_to_chunk_id = {}
    for cur_chunk_id in chunk_to_id_dict:
        shard_id = cv.meta.decode_chunk_position_number(cur_chunk_id)
        shard_filename = (
            str(cg.get_chunk_layer(cur_chunk_id)) + "/" + str(shard_id) + "-0.shard"
        )
        shard_to_chunk_id[shard_filename] = cur_chunk_id
        shard_filenames.append(shard_filename)
    mesh_dict = {}

    cf = CloudFiles(os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, "initial"))
    files_contents = cf.get(shard_filenames)
    for i in range(len(files_contents)):
        cur_chunk_id = shard_to_chunk_id[files_contents[i]["path"]]
        cur_layer = cg.get_chunk_layer(cur_chunk_id)
        if files_contents[i]["content"] is not None:
            disassembled_shard = cv.mesh.readers[cur_layer].disassemble_shard(
                files_contents[i]["content"]
            )
            nodes_in_chunk = chunk_to_id_dict[int(cur_chunk_id)]
            for node_in_chunk in nodes_in_chunk:
                node_in_chunk_int = int(node_in_chunk)
                if node_in_chunk_int in disassembled_shard:
                    mesh_dict[node_in_chunk_int] = disassembled_shard[node_in_chunk]
    del files_contents

    sharding_info = cv.mesh.meta.info["sharding"][str(layer)]
    sharding_spec = ShardingSpecification.from_dict(sharding_info)
    merged_meshes = {}
    biggest_frag = 0
    biggest_frag_vx_ct = 0
    bad_meshes = []
    number_frags_proc = 0

    # Process fragments in parallel using multiprocessing
    with ProcessPoolExecutor(
        initializer=_init_cg_worker, initargs=(cg.graph_id,)
    ) as executor:
        futures = {}
        for new_fragment_id, fragment_ids in multi_child_nodes.items():
            frag_mesh_data = {
                int(f): mesh_dict[int(f)] for f in fragment_ids if int(f) in mesh_dict
            }
            futures[
                executor.submit(
                    _process_sharded_fragment,
                    new_fragment_id,
                    fragment_ids,
                    frag_mesh_data,
                    layer,
                    mip,
                    chunk_id,
                    high_padding,
                )
            ] = new_fragment_id

        for future in as_completed(futures):
            frag_id, encoded_bytes, vertex_count, error = future.result()
            if encoded_bytes is not None:
                merged_meshes[int(frag_id)] = encoded_bytes
                number_frags_proc += 1
                if vertex_count > biggest_frag_vx_ct:
                    biggest_frag = frag_id
                    biggest_frag_vx_ct = vertex_count
                if number_frags_proc % 1000 == 0:
                    print(f"number frag proc = {number_frags_proc}")
            elif error is not None:
                print(error)
                bad_meshes.append(frag_id)

    del mesh_dict
    shard_binary = sharding_spec.synthesize_shard(merged_meshes)
    shard_filename = cv.mesh.readers[layer].get_filename(chunk_id)
    cf = CloudFiles(
        os.path.join(cv.cloudpath, cv.mesh.meta.mesh_path, "initial", str(layer))
    )
    cf.put(
        shard_filename,
        shard_binary,
        content_type="application/octet-stream",
        compress=False,
        cache_control=cache_string,
    )
    total_time = time.time() - start_existence_check_time

    ret = {
        "chunk_id": chunk_id,
        "total_time": total_time,
        "biggest_frag": biggest_frag,
        "biggest_frag_vx_ct": biggest_frag_vx_ct,
        "number_frag": number_frags_proc,
        "bad meshes": bad_meshes,
    }
    return ret
