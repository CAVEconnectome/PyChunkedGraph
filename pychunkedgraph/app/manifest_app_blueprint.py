from flask import Blueprint, request, make_response
from flask import current_app
# from google.cloud import pubsub_v1
import json
import numpy as np
import time
import datetime
import sys
import os
import traceback

from pychunkedgraph.meshing import meshgen
from pychunkedgraph.app import app_utils


bp = Blueprint('pychunkedgraph_manifest', __name__, url_prefix="/manifest")

# -------------------------------
# ------ Access control and index
# -------------------------------

@bp.route('/')
@bp.route("/index")
def index():
    return "Manifest Server -- 0.1"


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



@bp.route('/1.0/<node_id>/childmeshes', method=['POST', 'GET'])
def handle_children_meshes(node_id):
    cg = app_utils.get_cg()

    child = meshgen.get_downstream_multi_child_node(cg, node_id, stop_layer=2)

    return child
