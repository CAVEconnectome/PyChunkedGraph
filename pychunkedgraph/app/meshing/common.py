import json
import os

import numpy as np
import time
from datetime import datetime
import traceback
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


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


def before_request():
    current_app.request_start_time = time.time()
    current_app.request_start_date = datetime.utcnow()
    current_app.user_id = None
    current_app.table_id = None
    current_app.request_type = None


def after_request(response):
    dt = (time.time() - current_app.request_start_time) * 1000

    current_app.logger.debug("Response time: %.3fms" % dt)
    try:
        if current_app.user_id is None:
            user_id = ""
        else:
            user_id = current_app.user_id

        if current_app.table_id is not None:
            log_db = app_utils.get_log_db(current_app.table_id)
            log_db.add_success_log(
                user_id=user_id,
                user_ip="",
                request_time=current_app.request_start_date,
                response_time=dt,
                url=request.url,
                request_data=request.data,
                request_type=current_app.request_type,
            )
    except Exception as e:
        current_app.logger.debug(f"{current_app.user_id}: LogDB entry not"
                                 f" successful: {e}")

    return response


def unhandled_exception(e):
    status_code = 500
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": status_code,
            "traceback": tb,
        }
    )

    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": status_code,
        "message": str(e),
        "traceback": tb,
    }

    return jsonify(resp), status_code


def api_exception(e):
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": e.status_code.value,
            "traceback": tb,
        }
    )

    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": e.status_code.value,
        "message": str(e),
    }

    return jsonify(resp), e.status_code.value


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
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    verify = request.args.get("verify", False)
    verify = verify in ["True", "true", "1", True]

    cg = app_utils.get_cg(table_id)

    if "start_layer" in data:
        start_layer = int(data["start_layer"])
    else:
        start_layer = cg.get_chunk_layer(np.uint64(node_id))

    if "flexible_start_layer" in data:
        flexible_start_layer = int(data["flexible_start_layer"])
    else:
        flexible_start_layer = None

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg,
        np.uint64(node_id),
        stop_layer=2,
        start_layer=start_layer,
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
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id
    new_lvl2_ids = json.loads(request.data)["new_lvl2_ids"]
    
    with Connection(redis.from_url(current_app.config["REDIS_URL"])):
        
        if is_priority:
            retry=Retry(max=3, interval=[1, 10, 60])
            queue_name = "mesh-chunks"
        else:
            retry=Retry(max=3, interval=[60, 60, 60])
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