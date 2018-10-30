from flask import Blueprint, request, make_response, jsonify, Response
# from flask import current_app
# from google.cloud import pubsub_v1
import json
import numpy as np
# import time
# import datetime
# import sys
import os
# import traceback

from pychunkedgraph.meshing import meshgen, meshgen_utils
from pychunkedgraph.app import app_utils

# os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"

__version__ = '0.1.64'
bp = Blueprint('pychunkedgraph_meshing', __name__, url_prefix="/meshing")

# -------------------------------
# ------ Access control and index
# -------------------------------

@bp.route('/')
@bp.route("/index")
def index():
    return "Meshing Server -- " + __version__


@bp.route
def home():
    resp = make_response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


# ------------------------------------------------------------------------------


@bp.route('/1.0/<node_id>/mesh_preview', methods=['POST'])
def handle_preview_meshes(node_id):
    data = json.loads(request.data)
    node_id = np.uint64(node_id)

    cg = app_utils.get_cg()

    if "seg_ids" in data:
        seg_ids = data["seg_ids"]

        chunk_id = cg.get_chunk_id(node_id)
        supervoxel_ids = [cg.get_node_id(seg_id, chunk_id)
                          for seg_id in seg_ids]
    else:
        supervoxel_ids = None

    meshgen.mesh_lvl2_preview(cg, node_id, supervoxel_ids=supervoxel_ids,
                              cv_path=None, cv_mesh_dir=None, mip=2,
                              simplification_factor=999999,
                              max_err=40, parallel_download=8, verbose=True,
                              cache_control='no-cache')
    return Response(status=200)

@bp.route('/1.0/<node_id>/validfragments', methods=['POST', 'GET'])
def handle_valid_frags(node_id):
    cg = app_utils.get_cg()

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=1, verify_existence=True)

    return app_utils.tobinary(seg_ids)


@bp.route('/1.0/manifest/<node_id>:0', methods=['GET'])
def handle_get_manifest(node_id):
    # TODO: Read this from config
    MESH_MIP = 2

    verify = request.args.get('verify', False)
    verify = verify in ['True', 'true', '1', True]
    cg = app_utils.get_cg()
    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=2, verify_existence=verify)

    filenames = [meshgen_utils.get_mesh_name(cg, s, MESH_MIP) for s in seg_ids]

    return jsonify(fragments=filenames)
