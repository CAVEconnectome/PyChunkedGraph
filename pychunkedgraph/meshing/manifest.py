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

from pychunkedgraph.graph.utils.basetypes import NODE_ID  # noqa
from ..graph.types import empty_1d


def str_to_slice(slice_str: str):
    from re import match

    match = match(r"(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)", slice_str)
    return (
        slice(int(match.group(1)), int(match.group(2))),
        slice(int(match.group(3)), int(match.group(4))),
        slice(int(match.group(5)), int(match.group(6))),
    )


def slice_to_str(slices) -> str:
    if isinstance(slices, slice):
        return "%d-%d" % (slices.start, slices.stop)
    else:
        return "_".join(map(slice_to_str, slices))


def get_chunk_bbox(cg, chunk_id: np.uint64):
    layer = cg.get_chunk_layer(chunk_id)
    chunk_block_shape = get_mesh_block_shape(cg, layer)
    bbox_start = cg.get_chunk_coordinates(chunk_id) * chunk_block_shape
    bbox_end = bbox_start + chunk_block_shape
    return tuple(slice(bbox_start[i], bbox_end[i]) for i in range(3))


def get_chunk_bbox_str(cg, chunk_id: np.uint64) -> str:
    return slice_to_str(get_chunk_bbox(cg, chunk_id))


def get_mesh_name(cg, node_id: np.uint64) -> str:
    return f"{node_id}:0:{get_chunk_bbox_str(cg, node_id)}"


def get_mesh_block_shape(cg, graphlayer: int) -> np.ndarray:
    """
    Calculate the dimensions of a segmentation block that covers
    the same region as a ChunkedGraph chunk at layer `graphlayer`.
    """
    # Segmentation is not always uniformly downsampled in all directions.
    return cg.chunk_size * cg.fan_out ** np.max([0, graphlayer - 2])


def del_none_keys(d: dict):
    none_keys = []
    d_new = dict(d)
    for k, v in d.items():
        if v:
            continue
        none_keys.append(k)
        del d_new[k]
    return d_new, none_keys


def get_json_info(cg, mesh_dir: str = None):
    from json import loads, dumps

    dataset_info = cg.meta.dataset_info
    dummy_app_info = {"app": {"supported_api_versions": [0, 1]}}
    info = {**dataset_info, **dummy_app_info}
    if mesh_dir:
        info["mesh"] = mesh_dir

    info_str = dumps(info)
    return loads(info_str)


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
    cg,
    shard_readers,
    node_ids: Sequence[np.uint64],
    stop_layer: int = 2,
    mesh_dir: str = "graphene_meshes",
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
    with Storage(cg.cv_mesh_path) as stor:  # pylint: disable=not-context-manager
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
    from datetime import datetime

    if len(node_ids):
        node_ids = np.unique(node_ids)
    else:
        return {}, {}, []

    # initial_mesh_dt = np.datetime64(datetime(2020, 5, 10, 20, 50, 38, 934000))
    # node_ids_ts = cg.get_node_timestamps(node_ids)
    # initial_mesh_mask = node_ids_ts < initial_mesh_dt

    # initial_ids = node_ids[initial_mesh_mask]
    # new_ids = node_ids[~initial_mesh_mask]

    initial_ids = node_ids.copy()
    new_ids = np.array([])

    print("new_ids, initial_ids", new_ids.size, initial_ids.size)
    initial_meshes_d = _get_sharded_meshes(cg, shard_readers, initial_ids)
    new_meshes_d, missing_ids = _get_unsharded_meshes(cg, new_ids)
    return initial_meshes_d, new_meshes_d, missing_ids


def _get_mesh_paths(
    cg,
    node_ids: Sequence[np.uint64],
    stop_layer: int = 2,
    mesh_dir: str = "graphene_meshes",
) -> Dict:
    shard_readers = CloudVolume(  # pylint: disable=no-member
        f"graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=mesh_dir,
        info=get_json_info(cg, mesh_dir=mesh_dir),
    ).mesh

    result = {}
    node_layers = cg.get_chunk_layers(node_ids)
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
    # UNIX_TIMESTAMP = 1562100638 # {"iso":"2019-07-02 20:50:38.934000+00:00"}
    MAX_STITCH_LAYER = cg.meta.custom_data.get("mesh", {}).get("max_layer", 6)

    start = time()
    node_ids = _get_children_before_start_layer(cg, node_id, MAX_STITCH_LAYER)
    print("_get_children_before_start_layer: %.3fs" % (time() - start))
    print("node_ids", len(node_ids))

    start = time()
    result = _get_mesh_paths(cg, node_ids)
    node_ids = np.fromiter(result.keys(), dtype=NODE_ID)

    mesh_files = []
    for val in result.values():
        try:
            path, offset, size = val
            path = path.split("initial/")[-1]
            mesh_files.append(f"~{path}:{offset}:{size}")
        except:
            mesh_files.append(val)
    print("shard lookups took: %.3fs" % (time() - start))
    return node_ids, mesh_files


def speculative_manifest(
    cg, node_id, stop_layer: int = 2, mesh_dir: str = "graphene_meshes"
):
    """
    This assumes children IDs have meshes.
    Not checking for their existence reduces latency.
    """
    # UNIX_TIMESTAMP = 1562100638 # {"iso":"2019-07-02 20:50:38.934000+00:00"}
    MAX_STITCH_LAYER = cg.meta.custom_data.get("mesh", {}).get("max_layer", 2)

    start = time()
    node_ids = _get_children_before_start_layer(cg, node_id, MAX_STITCH_LAYER)
    print("_get_children_before_start_layer", time() - start)

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
        mesh_dir=mesh_dir,
        info=get_json_info(cg, mesh_dir=mesh_dir),
    ).mesh.readers

    node_ids = np.concatenate(result)
    layers = cg.get_chunk_layers(node_ids)
    chunk_ids = cg.get_chunk_ids_from_node_ids(node_ids)
    fragment_URIs = []
    for id_, layer, chunk_id in zip(node_ids, layers, chunk_ids):
        fname, minishard = readers[layer].compute_shard_location(id_)
        fragment_URIs.append(f"{id_}:{layer}:{chunk_id}:{fname}:{minishard}")
    return fragment_URIs
