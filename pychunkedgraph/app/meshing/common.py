# pylint: disable=invalid-name, missing-docstring, unspecified-encoding, assigning-non-slot

import json
import os
from threading import Thread

import numpy as np
import redis
from rq import Queue, Connection, Retry
from flask import Response, current_app, g, jsonify, make_response, request

from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.app.meshing import tasks as meshing_tasks
from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.meshing import meshgen, meshgen_utils


# -------------------------------
# ------ Access control and index
# -------------------------------

__meshing_url_prefix__ = os.environ.get('MESHING_URL_PREFIX', 'meshing')

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

    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    cg = app_utils.get_cg(table_id)

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=1, verify_existence=True
    )

    return app_utils.tobinary(seg_ids)


## MANIFEST --------------------------------------------------------------------


def handle_get_manifest(table_id, node_id):
    current_app.request_type = "manifest"
    current_app.table_id = table_id

    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    if len(request.data) > 0:
        data = json.loads(request.data)
    else:
        data = {}

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=int
        ).T
    else:
        bounding_box = None

    verify = request.args.get("verify", False)
    verify = verify in ["True", "true", "1", True]

    cg = app_utils.get_cg(table_id)
    if "flexible_start_layer" in data:
        flexible_start_layer = int(data["flexible_start_layer"])
    else:
        flexible_start_layer = None

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg,
        np.uint64(node_id),
        stop_layer=2,
        bounding_box=bounding_box,
        verify_existence=verify,
        flexible_start_layer=flexible_start_layer
    )

    filenames = [meshgen_utils.get_mesh_name(cg, s) for s in seg_ids]

    resp = {
        "fragments": filenames
    }

    if "return_seg_id_layers" in data:
        if app_utils.toboolean(data["return_seg_id_layers"]):
            resp["seg_id_layers"] = cg.get_chunk_layers(seg_ids)

    if "return_seg_chunk_coordinates" in data:
        if app_utils.toboolean(data["return_seg_chunk_coordinates"]):
            resp["seg_chunk_coordinates"] = [cg.get_chunk_coordinates(seg_id)
                                             for seg_id in seg_ids]

    return resp

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

## REMESHING -----------------------------------------------------
def handle_remesh(table_id):
    current_app.request_type = "remesh_enque"
    current_app.table_id = table_id
    is_priority = request.args.get('priority', True, type=str2bool)
    is_redisjob = request.args.get('use_redis', False, type=str2bool)
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id
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
            task = q.enqueue(meshing_tasks.remeshing, table_id,
                             new_lvl2_ids)

        response_object = {
            "status": "success",
            "data": {
                "task_id": task.get_id()
            }
        }

        return jsonify(response_object), 202
    else:
        new_lvl2_ids = np.array(new_lvl2_ids, dtype=np.uint64)
        cg = app_utils.get_cg(table_id)

        if len(new_lvl2_ids) > 0:
            t = Thread(
                target=_remeshing,
                args=(cg.get_serialized_info(), new_lvl2_ids)
            )
            t.start()

        return Response(status=202)


def _remeshing(serialized_cg_info, lvl2_nodes):
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg, lvl2_nodes, stop_layer=4, mesh_path=None, mip=1,
        max_err=320
    )

    return Response(status=200)
