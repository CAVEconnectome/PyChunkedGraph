from flask import Blueprint
from middle_auth_client import auth_requires_permission

from pychunkedgraph.app.meshing import common
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

bp = Blueprint("pcg_meshing_v0", __name__, url_prefix="/meshing/1.0")

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


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


@bp.before_request
def before_request():
    return common.before_request()


@bp.after_request
def after_request(response):
    return common.after_request(response)


@bp.errorhandler(Exception)
def unhandled_exception(e):
    return common.unhandled_exception(e)


@bp.errorhandler(cg_exceptions.ChunkedGraphAPIError)
def api_exception(e):
    return common.api_exception(e)


## VALIDFRAGMENTS --------------------------------------------------------------


@bp.route("/<table_id>/<node_id>/validfragments", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_valid_frags(table_id, node_id):
    return common.handle_valid_frags(table_id, node_id)


## MANIFEST --------------------------------------------------------------------


@bp.route("/<table_id>/manifest/<node_id>:0", methods=["GET"])
@auth_requires_permission("view")
def handle_get_manifest(table_id, node_id):
    return common.handle_get_manifest(table_id, node_id)
