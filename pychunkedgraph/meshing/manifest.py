import multiprocessing as mp
from time import time
from typing import List
from typing import Dict
from typing import Tuple
from typing import Sequence
from functools import lru_cache

import numpy as np
from cloudvolume import CloudVolume, Storage
from multiwrapper import multiprocessing_utils as mu

from .meshgen_utils import get_mesh_name
from .meshgen_utils import get_json_info
from ..graph.types import empty_1d
from ..graph.utils.basetypes import NODE_ID


def get_highest_child_nodes_with_meshes(
    cg,
    node_id: np.uint64,
    stop_layer=2,
    start_layer=None,
    verify_existence=False,
    bounding_box=None,
    flexible_start_layer=None,
):
    # if not cg.sharded_meshes:
    #     return children_meshes_non_sharded(
    #         cg,
    #         node_id,
    #         stop_layer=stop_layer,
    #         start_layer=start_layer,
    #         verify_existence=verify_existence,
    #         bounding_box=bounding_box,
    #         flexible_start_layer=flexible_start_layer,
    #     )
    return children_meshes_sharded(
        cg,
        node_id,
        stop_layer=stop_layer,
        verify_existence=verify_existence,
        bounding_box=bounding_box,
    )


def children_meshes_non_sharded(
    cg,
    node_id: np.uint64,
    stop_layer=2,
    start_layer=None,
    verify_existence=False,
    bounding_box=None,
    flexible_start_layer=None,
):
    if flexible_start_layer is not None:
        # Get highest children that are at flexible_start_layer or below
        # (do this because of skip connections)
        candidates = cg.get_children_at_layer(node_id, flexible_start_layer, True)
    elif start_layer is None:
        candidates = np.array([node_id], dtype=np.uint64)
    else:
        candidates = cg.get_subgraph(
            node_id,
            bbox=bounding_box,
            bbox_is_coordinate=True,
            return_layers=[start_layer],
            nodes_only=True,
        )

    if verify_existence:
        valid_node_ids = []
        with Storage(cg.cv_mesh_path) as stor:  # pylint: disable=not-context-manager
            while True:
                filenames = [get_mesh_name(cg, c) for c in candidates]

                start = time()
                existence_dict = stor.files_exist(filenames)
                print("Existence took: %.3fs" % (time() - start))

                missing_meshes = []
                for mesh_key in existence_dict:
                    node_id = np.uint64(mesh_key.split(":")[0])
                    if existence_dict[mesh_key]:
                        valid_node_ids.append(node_id)
                    else:
                        if cg.get_chunk_layer(node_id) > stop_layer:
                            missing_meshes.append(node_id)

                start = time()
                if missing_meshes:
                    candidates = cg.get_children(missing_meshes, flatten=True)
                else:
                    break
                print("ChunkedGraph lookup took: %.3fs" % (time() - start))
    else:
        valid_node_ids = candidates
    return valid_node_ids, [get_mesh_name(cg, s) for s in valid_node_ids]


def del_none_keys(d: dict):
    none_keys = []
    d_new = dict(d)
    for k, v in d.items():
        if v:
            continue
        none_keys.append(k)
        del d_new[k]
    return d_new, none_keys


def _segregate_node_ids(cg, node_ids):
    from datetime import datetime

    initial_mesh_dt = np.datetime64(
        datetime.fromtimestamp(
            cg.meta.custom_data.get("mesh", {}).get("initial_ts", datetime.now())
        )
    )
    node_ids_ts = cg.get_node_timestamps(node_ids)
    initial_mesh_mask = node_ids_ts < initial_mesh_dt
    initial_ids = node_ids[initial_mesh_mask]
    new_ids = node_ids[~initial_mesh_mask]
    return initial_ids, new_ids


def _get_children(cg, node_ids: Sequence[np.uint64], children_cache: dict = {}):
    if not len(node_ids):
        return empty_1d.copy()
    node_ids = np.array(node_ids, dtype=NODE_ID)
    mask = np.in1d(node_ids, np.fromiter(children_cache.keys(), dtype=NODE_ID))
    children_d = cg.get_children(node_ids[~mask])
    children_cache.update(children_d)

    children = [empty_1d]
    for id_ in node_ids:
        children.append(children_cache[id_])
    return np.concatenate(children)


def _check_skips(cg, node_ids: Sequence[np.uint64], children_cache: dict = {}):
    """
    If a node ID has a single child, it is considered a skip.
    Such IDs won't have meshes because the child mesh will be identical.
    """
    start = time()
    layers = cg.get_chunk_layers(node_ids)
    skips = []
    result = [empty_1d, node_ids[layers == 2]]
    children_d = cg.get_children(node_ids[layers > 2])
    for p, c in children_d.items():
        if c.size > 1:
            result.append([p])
            children_cache[p] = c
            continue
        assert c.size == 1, f"{p} does not seem to have children."
        skips.append(c[0])
    print(f"skips {len(skips)}, total {len(node_ids)}, time {time()-start}")
    return np.concatenate(result), np.array(skips, dtype=np.uint64)


def _get_sharded_meshes(
    cg, shard_readers, node_ids: Sequence[np.uint64], stop_layer: int = 2,
) -> Dict:
    children_cache = {}
    result = {}
    if not len(node_ids):
        return result
    node_layers = cg.get_chunk_layers(node_ids)
    l2_ids = [node_ids[node_layers == stop_layer]]
    while np.any(node_layers > stop_layer):
        l2_ids.append(node_ids[node_layers == stop_layer])
        ids_ = node_ids[node_layers > stop_layer]
        ids_, skips = _check_skips(cg, ids_, children_cache=children_cache)

        start = time()
        result_ = shard_readers.initial_exists(ids_, return_byte_range=True)
        result_, missing_ids = del_none_keys(result_)
        result.update(result_)
        print("ids, missing", ids_.size, len(missing_ids), time() - start)

        # node_ids = cg.get_children(missing_ids, flatten=True)
        node_ids = _get_children(cg, missing_ids, children_cache=children_cache)
        node_ids = np.concatenate([node_ids, skips])
        node_layers = cg.get_chunk_layers(node_ids)

    # remainder IDs
    start = time()
    l2_ids = np.concatenate([*l2_ids, node_ids[node_layers == stop_layer]])
    # result_ = shard_readers.readers[stop_layer].exists(
    #     labels=l2_ids, path=f"{mesh_dir}/initial/{stop_layer}/", return_byte_range=True,
    # )
    result_ = shard_readers.initial_exists(l2_ids, return_byte_range=True)
    print(f"{stop_layer}:{l2_ids.size} {time()-start}")
    result_, temp = del_none_keys(result_)
    print("missing_ids", len(temp))
    print(temp)
    result.update(result_)
    return result


def _get_unsharded_meshes(cg, node_ids: Sequence[np.uint64]) -> Tuple[Dict, List]:
    result = {}
    missing_ids = []
    if not len(node_ids):
        return result, missing_ids
    mesh_dir = cg.meta.custom_data.get("mesh", {}).get("dir", "graphene_meshes")
    mesh_path = f"{cg.meta.data_source.WATERSHED}/{mesh_dir}/dynamic"
    with Storage(mesh_path) as stor:  # pylint: disable=not-context-manager
        filenames = [get_mesh_name(cg, c) for c in node_ids]
        start = time()
        existence_dict = stor.files_exist(filenames)
        print("bucket existence took: %.3fs" % (time() - start))

        for mesh_key in existence_dict:
            node_id = np.uint64(mesh_key.split(":")[0])
            if existence_dict[mesh_key]:
                result[node_id] = mesh_key
                continue
            missing_ids.append(node_id)
    missing_ids = np.array(missing_ids, dtype=NODE_ID)
    return result, missing_ids


def _get_sharded_unsharded_meshes(
    cg, shard_readers: Dict, node_ids: Sequence[np.uint64],
) -> Tuple[Dict, Dict, List]:
    if len(node_ids):
        node_ids = np.unique(node_ids)
    else:
        return {}, {}, []

    initial_ids, new_ids = _segregate_node_ids(cg, node_ids)
    print("new_ids, initial_ids", new_ids.size, initial_ids.size)
    initial_meshes_d = _get_sharded_meshes(cg, shard_readers, initial_ids)
    new_meshes_d, missing_ids = _get_unsharded_meshes(cg, new_ids)
    return initial_meshes_d, new_meshes_d, missing_ids


def _get_mesh_paths(cg, node_ids: Sequence[np.uint64], stop_layer: int = 2,) -> Dict:
    shard_readers = CloudVolume(  # pylint: disable=no-member
        f"graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=cg.meta.custom_data.get("mesh", {}).get("dir", "graphene_meshes"),
        info=get_json_info(cg),
    ).mesh

    result = {}
    node_layers = cg.get_chunk_layers(node_ids)
    node_ids = node_ids[node_layers > 1]
    while np.any(node_layers > stop_layer):
        resp = _get_sharded_unsharded_meshes(cg, shard_readers, node_ids)
        initial_meshes_d, new_meshes_d, missing_ids = resp
        result.update(initial_meshes_d)
        result.update(new_meshes_d)
        node_ids = cg.get_children(missing_ids, flatten=True)
        node_layers = cg.get_chunk_layers(node_ids)

    # check for left over level 2 IDs
    print("node_ids left over", node_ids.size)
    resp = _get_sharded_unsharded_meshes(cg, shard_readers, node_ids)
    initial_meshes_d, new_meshes_d, _ = resp
    result.update(initial_meshes_d)
    result.update(new_meshes_d)
    return result


def _get_children_before_start_layer(cg, node_id: np.uint64, start_layer: int = 6):
    if cg.get_chunk_layer(node_id) == 2:
        return np.array([node_id], dtype=NODE_ID)
    result = [empty_1d]
    parents = np.array([node_id], dtype=np.uint64)
    while parents.size:
        children = cg.get_children(parents, flatten=True)
        layers = cg.get_chunk_layers(children)
        result.append(children[layers <= start_layer])
        parents = children[layers > start_layer]
    return np.concatenate(result)


def children_meshes_sharded(
    cg, node_id: np.uint64, stop_layer=2, verify_existence=False, bounding_box=None,
):
    """
    For each ID, first check for new meshes,
    If not found check initial meshes.
    """
    MAX_STITCH_LAYER = cg.meta.custom_data.get("mesh", {}).get("max_layer", 2)
    start = time()
    node_ids = _get_children_before_start_layer(cg, node_id, MAX_STITCH_LAYER)
    print(f"children_before_start_layer {time() - start}, count {len(node_ids)}")

    start = time()
    result = _get_mesh_paths(cg, node_ids)
    node_ids = np.fromiter(result.keys(), dtype=NODE_ID)

    mesh_files = []
    for val in result.values():
        try:
            path, offset, size = val
            path = path.split("initial/")[-1]
            # TODO change this to include node ID in response
            mesh_files.append(f"~{path}:{offset}:{size}")
        except:
            mesh_files.append(val)
    print("shard lookups took: %.3fs" % (time() - start))
    return node_ids, mesh_files


def speculative_manifest(cg, node_id, stop_layer: int = 2):
    """
    This assumes children IDs have meshes.
    Not checking for their existence reduces latency.
    """
    MAX_STITCH_LAYER = cg.meta.custom_data.get("mesh", {}).get("max_layer", 2)

    start = time()
    node_ids = _get_children_before_start_layer(cg, node_id, MAX_STITCH_LAYER)
    print("children_before_start_layer", time() - start)

    start = time()
    result = [empty_1d]
    node_layers = cg.get_chunk_layers(node_ids)
    while np.any(node_layers > stop_layer):
        result.append(node_ids[node_layers == stop_layer])
        ids_ = node_ids[node_layers > stop_layer]
        ids_, skips = _check_skips(cg, ids_)

        result.append(ids_)
        node_ids = skips.copy()
        node_layers = cg.get_chunk_layers(node_ids)

    result.append(node_ids[node_layers == stop_layer])
    print("chilren IDs", len(result), time() - start)

    readers = CloudVolume(  # pylint: disable=no-member
        f"graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=cg.meta.custom_data.get("mesh", {}).get("dir", "graphene_meshes"),
        info=get_json_info(cg),
    ).mesh.readers

    node_ids = np.concatenate(result)
    initial_ids, new_ids = _segregate_node_ids(cg, node_ids)

    # get shards for initial IDs
    layers = cg.get_chunk_layers(initial_ids)
    chunk_ids = cg.get_chunk_ids_from_node_ids(initial_ids)
    mesh_shards = []
    for id_, layer, chunk_id in zip(initial_ids, layers, chunk_ids):
        fname, minishard = readers[layer].compute_shard_location(id_)
        mesh_shards.append(f"~{id_}:{layer}:{chunk_id}:{fname}:{minishard}")

    # get mesh files for new IDs
    mesh_files = [f"{get_mesh_name(cg, id_)}" for id_ in new_ids]
    return np.concatenate([initial_ids, new_ids]), mesh_shards + mesh_files
