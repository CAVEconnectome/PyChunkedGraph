import json
import os

import numpy as np
import time
from datetime import datetime
import traceback
from flask import Response, current_app, g, jsonify, make_response, request

from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.graph import chunkedgraph
from pychunkedgraph.meshing import meshgen, meshgen_utils


# -------------------------------
# ------ Access control and index
# -------------------------------

__meshing_url_prefix__ = os.environ.get("MESHING_URL_PREFIX", "meshing")


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


def _remeshing(serialized_cg_info, lvl2_nodes):
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)
    cv_mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    cv_unsharded_mesh_path = os.path.join(cg.meta.data_source.WATERSHED, cv_mesh_dir, cv_unsharded_mesh_dir)

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg, lvl2_nodes, stop_layer=6, mip=2, max_err=40,
        cv_sharded_mesh_dir=cv_mesh_dir, cv_unsharded_mesh_path=cv_unsharded_mesh_path
    )

    return Response(status=200)


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
        current_app.logger.debug(
            f"{current_app.user_id}: LogDB entry not" f" successful: {e}"
        )

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

    return_seg_ids = request.args.get("return_seg_ids", False)
    return_seg_ids = return_seg_ids in ["True", "true", "1", True]

    cg = app_utils.get_cg(table_id)

    if "start_layer" in data:
        start_layer = int(data["start_layer"])
    else:
        start_layer = cg.get_chunk_layer(np.uint64(node_id))

    if "flexible_start_layer" in data:
        flexible_start_layer = int(data["flexible_start_layer"])
    else:
        flexible_start_layer = None

    seg_ids, fragment_URIs = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg,
        np.uint64(node_id),
        stop_layer=2,
        start_layer=start_layer,
        bounding_box=bounding_box,
        verify_existence=verify,
        flexible_start_layer=flexible_start_layer,
    )

    resp = {"fragments": fragment_URIs}

    if "return_seg_id_layers" in data:
        if app_utils.toboolean(data["return_seg_id_layers"]):
            resp["seg_id_layers"] = cg.get_chunk_layers(seg_ids)

    if "return_seg_chunk_coordinates" in data:
        if app_utils.toboolean(data["return_seg_chunk_coordinates"]):
            resp["seg_chunk_coordinates"] = [
                cg.get_chunk_coordinates(seg_id) for seg_id in seg_ids
            ]
    if return_seg_ids:
        resp["seg_ids"] = seg_ids
    return resp
