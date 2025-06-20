# pylint: disable=invalid-name, missing-docstring
import json
import os
import threading

import numpy as np
import redis
from rq import Queue, Connection, Retry
from flask import Response, current_app, jsonify, make_response, request

from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.graph import chunkedgraph
from pychunkedgraph.app.meshing import tasks as meshing_tasks
from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing.manifest import get_highest_child_nodes_with_meshes
from pychunkedgraph.meshing.manifest import get_children_before_start_layer
from pychunkedgraph.meshing.manifest import ManifestCache


__meshing_url_prefix__ = os.environ.get("MESHING_URL_PREFIX", "meshing")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def index():
    return f"PyChunkedGraph Meshing v{__version__}"


def home():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


## VALIDFRAGMENTS --------------------------------------------------------------


def handle_valid_frags(table_id, node_id):
    current_app.table_id = table_id
    cg = app_utils.get_cg(table_id)
    seg_ids = get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=1, verify_existence=True
    )
    return app_utils.tobinary(seg_ids)


## MANIFEST --------------------------------------------------------------------


def handle_get_manifest(table_id, node_id):
    current_app.request_type = "manifest"
    current_app.table_id = table_id

    data = {}
    if len(request.data) > 0:
        data = json.loads(request.data)

    bounding_box = None
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")], dtype=int).T

    cg = app_utils.get_cg(table_id)
    verify = request.args.get("verify", False)
    verify = verify in ["True", "true", "1", True]
    return_seg_ids = request.args.get("return_seg_ids", False)
    prepend_seg_ids = request.args.get("prepend_seg_ids", False)
    return_seg_ids = return_seg_ids in ["True", "true", "1", True]
    prepend_seg_ids = prepend_seg_ids in ["True", "true", "1", True]
    start_layer = cg.meta.custom_data.get("mesh", {}).get("max_layer", 2)
    start_layer = int(request.args.get("start_layer", start_layer))
    if "start_layer" in data:
        start_layer = int(data["start_layer"])

    flexible_start_layer = None
    if "flexible_start_layer" in data:
        flexible_start_layer = int(data["flexible_start_layer"])
    args = (
        node_id,
        verify,
        return_seg_ids,
        prepend_seg_ids,
        start_layer,
        flexible_start_layer,
        bounding_box,
        data,
    )
    return manifest_response(cg, args)


def manifest_response(cg, args):
    from pychunkedgraph.meshing.manifest import speculative_manifest_sharded

    (
        node_id,
        verify,
        return_seg_ids,
        prepend_seg_ids,
        start_layer,
        flexible_start_layer,
        bounding_box,
        data,
    ) = args
    resp = {}
    seg_ids = []
    if not verify:
        seg_ids, resp["fragments"] = speculative_manifest_sharded(
            cg, node_id, start_layer=start_layer, bounding_box=bounding_box
        )

    else:
        seg_ids, resp["fragments"] = get_highest_child_nodes_with_meshes(
            cg,
            np.uint64(node_id),
            start_layer=start_layer,
            bounding_box=bounding_box,
        )
    if prepend_seg_ids:
        resp["fragments"] = [f"~{i}:{f}" for i, f in zip(seg_ids, resp["fragments"])]
    if return_seg_ids:
        resp["seg_ids"] = seg_ids
    return _check_post_options(cg, resp, data, seg_ids)


def _check_post_options(cg, resp, data, seg_ids):
    if app_utils.toboolean(data.get("return_seg_ids", "false")):
        resp["seg_ids"] = seg_ids
    if app_utils.toboolean(data.get("return_seg_id_layers", "false")):
        resp["seg_id_layers"] = cg.get_chunk_layers(seg_ids)
    if app_utils.toboolean(data.get("return_seg_chunk_coordinates", "false")):
        resp["seg_chunk_coordinates"] = [
            cg.get_chunk_coordinates(seg_id) for seg_id in seg_ids
        ]
    return resp


## REMESHING -----------------------------------------------------
def handle_remesh(table_id):
    current_app.request_type = "remesh_enque"
    current_app.table_id = table_id
    is_priority = request.args.get("priority", True, type=str2bool)
    is_redisjob = request.args.get("use_redis", False, type=str2bool)

    new_lvl2_ids = json.loads(request.data)["new_lvl2_ids"]

    if is_redisjob:
        with Connection(redis.from_url(current_app.config["REDIS_URL"])):

            if is_priority:
                retry = Retry(max=3, interval=[1, 10, 60])
                queue_name = "mesh-chunks"
            else:
                retry = Retry(max=3, interval=[60, 60, 60])
                queue_name = "mesh-chunks-low-priority"
            q = Queue(queue_name, retry=retry, default_timeout=1200)
            task = q.enqueue(meshing_tasks.remeshing, table_id, new_lvl2_ids)

        response_object = {"status": "success", "data": {"task_id": task.get_id()}}

        return jsonify(response_object), 202
    else:
        new_lvl2_ids = np.array(new_lvl2_ids, dtype=np.uint64)
        cg = app_utils.get_cg(table_id)

        if len(new_lvl2_ids) > 0:
            t = threading.Thread(
                target=_remeshing, args=(cg.get_serialized_info(), new_lvl2_ids)
            )
            t.start()

        return Response(status=202)


def _remeshing(serialized_cg_info, lvl2_nodes):
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)
    cv_mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    cv_unsharded_mesh_path = os.path.join(
        cg.meta.data_source.WATERSHED, cv_mesh_dir, cv_unsharded_mesh_dir
    )
    mesh_data = cg.meta.custom_data["mesh"]

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg,
        lvl2_nodes,
        stop_layer=mesh_data["max_layer"],
        mip=mesh_data["mip"],
        max_err=mesh_data["max_error"],
        cv_sharded_mesh_dir=cv_mesh_dir,
        cv_unsharded_mesh_path=cv_unsharded_mesh_path,
    )

    return Response(status=200)


def clear_manifest_cache(cg, node_id):
    node_ids = get_children_before_start_layer(cg, node_id, start_layer=2)
    ManifestCache(cg.graph_id).clear_fragments(node_ids)
