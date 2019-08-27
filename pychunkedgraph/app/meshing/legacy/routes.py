import json

import numpy as np
from flask import Blueprint, Response, request

from pychunkedgraph.app import app_utils
from pychunkedgraph.app.meshing import common
from pychunkedgraph.meshing import meshgen

bp = Blueprint("pcg_meshing_v0", __name__, url_prefix="/meshing/")

# -------------------------------
# ------ Access control and index
# -------------------------------


@bp.route("/")
@bp.route("/index")
def index():
    return common.index()


@bp.route
def home():
    return common.home()


# ------------------------------------------------------------------------------


@bp.route("/1.0/<table_id>/<node_id>/mesh_preview", methods=["POST", "GET"])
def handle_preview_meshes(table_id, node_id):
    if len(request.data) > 0:
        data = json.loads(request.data)
    else:
        data = {}

    node_id = np.uint64(node_id)

    cg = app_utils.get_cg(table_id)

    if "seg_ids" in data:
        seg_ids = data["seg_ids"]

        chunk_id = cg.get_chunk_id(node_id)
        supervoxel_ids = [cg.get_node_id(seg_id, chunk_id) for seg_id in seg_ids]
    else:
        supervoxel_ids = None

    meshgen.mesh_lvl2_preview(
        cg,
        node_id,
        supervoxel_ids=supervoxel_ids,
        cv_path=None,
        cv_mesh_dir=None,
        mip=2,
        simplification_factor=999999,
        max_err=40,
        parallel_download=1,
        verbose=True,
        cache_control="no-cache",
    )
    return Response(status=200)


## VALIDFRAGMENTS --------------------------------------------------------------


@bp.route("/1.0/<table_id>/<node_id>/validfragments", methods=["POST", "GET"])
def handle_valid_frags(table_id, node_id):
    return common.handle_valid_frags(table_id, node_id)


## MANIFEST --------------------------------------------------------------------


@bp.route("/1.0/<table_id>/manifest/<node_id>:0", methods=["GET"])
def handle_get_manifest(table_id, node_id):
    return common.handle_get_manifest(table_id, node_id)
