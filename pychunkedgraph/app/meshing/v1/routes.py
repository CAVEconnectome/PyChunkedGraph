# pylint: disable=invalid-name, missing-docstring

from flask import Blueprint
from middle_auth_client import auth_requires_permission, auth_required

from pychunkedgraph.app import common as app_common
from pychunkedgraph.app.meshing import common
from pychunkedgraph.graph import exceptions as cg_exceptions
from pychunkedgraph.app.app_utils import get_cg
from pychunkedgraph.app.app_utils import remap_public


bp = Blueprint(
    "pcg_meshing_v1", __name__, url_prefix=f"/{common.__meshing_url_prefix__}/api/v1"
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
# @auth_required
def before_request():
    return app_common.before_request()


@bp.after_request
# @auth_required
def after_request(response):
    return app_common.after_request(response)


@bp.errorhandler(Exception)
def unhandled_exception(e):
    return app_common.unhandled_exception(e)


@bp.errorhandler(cg_exceptions.ChunkedGraphAPIError)
def api_exception(e):
    return app_common.api_exception(e)


## VALIDFRAGMENTS --------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/validfragments", methods=["GET"])
@auth_requires_permission("view")
@remap_public
def handle_valid_frags(table_id, node_id):
    return common.handle_valid_frags(table_id, node_id)


## MANIFEST --------------------------------------------------------------------


@bp.route("/table/<table_id>/manifest/<node_id>:0", methods=["GET"])
@auth_requires_permission(
    "view",
    public_table_key="table_id",
    public_node_key="node_id",
)
@remap_public
def handle_get_manifest(table_id, node_id):
    return common.handle_get_manifest(table_id, node_id)


@bp.route("/table/<table_id>/manifest/multiscale/<node_id>", methods=["GET"])
@auth_requires_permission(
    "view",
    public_table_key="table_id",
    public_node_key="node_id",
)
@remap_public
def handle_get_multilod_manifest(table_id, node_id):
    return common.handle_get_manifest(table_id, node_id, multiscale=True)


## ENQUE MESHING JOBS ----------------------------------------------------------


@bp.route("/table/<table_id>/remeshing", methods=["POST"])
@auth_requires_permission("edit")
@remap_public(edit=True)
def handle_remesh(table_id):
    return common.handle_remesh(table_id)


@bp.route("/table/<table_id>/clear_manifest_cache/<node_id>", methods=["POST"])
@auth_requires_permission("admin")
def handle_clear_manifest_cache(table_id, node_id):
    cg = get_cg(table_id)
    common.clear_manifest_cache(cg, node_id)
