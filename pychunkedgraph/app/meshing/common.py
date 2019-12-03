import json

import numpy as np
from flask import Response, jsonify, make_response, request

from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.meshing import meshgen, meshgen_utils


# -------------------------------
# ------ Access control and index
# -------------------------------


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

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg, lvl2_nodes, stop_layer=4, cv_path=None, cv_mesh_dir=None, mip=1, max_err=320
    )

    return Response(status=200)


## VALIDFRAGMENTS --------------------------------------------------------------


def handle_valid_frags(table_id, node_id):
    cg = app_utils.get_cg(table_id)

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=1, verify_existence=True
    )

    return app_utils.tobinary(seg_ids)


## MANIFEST --------------------------------------------------------------------


def handle_get_manifest(table_id, node_id):
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

    if "return_seg_id_layers" in data:
        if app_utils.toboolean(data["return_seg_id_layers"]):
            return jsonify(fragments=filenames, seg_id_layers=cg.get_chunk_layers(seg_ids))

    return jsonify(fragments=filenames)
