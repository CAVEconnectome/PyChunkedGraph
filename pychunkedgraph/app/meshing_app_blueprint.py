from flask import Blueprint, request, make_response, jsonify
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

__version__ = '0.1.42'
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


@bp.route('/1.0/mesh_lvl', methods=['POST'])
def mesh_():
    data = json.loads(request.data)

    assert "node_id" in data
    node_id = data["node_id"]

    if "sv_ids" in data:
        sv_ids = data["sv_ids"]
    else:
        sv_ids = None

    cg = app_utils.get_cg()

    meshgen.mesh_lvl2_preview(cg, node_id, mip=2, supervoxel_ids=sv_ids)


@bp.route('/1.0/<node_id>/validfragments', methods=['POST', 'GET'])
def handle_valid_frags(node_id):
    cg = app_utils.get_cg()

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(cg,
                                                                np.uint64(node_id),
                                                                stop_layer=1)

    return app_utils.tobinary(seg_ids)
