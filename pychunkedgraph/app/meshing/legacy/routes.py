# pylint: disable=invalid-name, missing-docstring, unspecified-encoding, assigning-non-slot

from flask import Blueprint
from middle_auth_client import auth_requires_permission, auth_required

from pychunkedgraph.app import common as app_common
from pychunkedgraph.app.meshing import common
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.app.app_utils import remap_public

bp = Blueprint(
    "pcg_meshing_v0", __name__, url_prefix=f"/{common.__meshing_url_prefix__}/1.0"
)

# -------------------------------
# ------ Access control and index
# -------------------------------


@bp.route("/")
@bp.route("/index")
@auth_required
def index():
    return common.index()


@bp.route
@auth_required
def home():
    return common.home()


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


@bp.before_request
@auth_required
def before_request():
    return app_common.before_request()


@bp.after_request
def after_request(response):
    return app_common.after_request(response)


@bp.errorhandler(Exception)
def unhandled_exception(e):
    return app_common.unhandled_exception(e)


@bp.errorhandler(cg_exceptions.ChunkedGraphAPIError)
def api_exception(e):
    return app_common.api_exception(e)


## VALIDFRAGMENTS --------------------------------------------------------------


@bp.route("/<table_id>/<node_id>/validfragments", methods=["POST", "GET"])
@auth_requires_permission("view")
@remap_public
def handle_valid_frags(table_id, node_id):
    return common.handle_valid_frags(table_id, node_id)


## MANIFEST --------------------------------------------------------------------


@bp.route("/<table_id>/manifest/<node_id>:0", methods=["GET"])
@auth_requires_permission("view")
@remap_public
def handle_get_manifest(table_id, node_id):
    return common.handle_get_manifest(table_id, node_id)
