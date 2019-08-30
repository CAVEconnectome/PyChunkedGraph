from flask import Blueprint, Response

from pychunkedgraph.app.meshing import common

bp = Blueprint("pcg_meshing_v1", __name__, url_prefix="/meshing/api/v1/")

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


@bp.route("/table/<table_id>/node/<node_id>/mesh_preview", methods=["POST", "GET"])
def handle_preview_meshes(table_id, node_id):  # pylint: disable=unused-argument
    return Response(status=410)


## VALIDFRAGMENTS --------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/validfragments", methods=["GET"])
def handle_valid_frags(table_id, node_id):
    return common.handle_valid_frags(table_id, node_id)


## MANIFEST --------------------------------------------------------------------


@bp.route("/table/<table_id>/manifest/<node_id>:0", methods=["GET"])
def handle_get_manifest(table_id, node_id):
    return common.handle_get_manifest(table_id, node_id)
