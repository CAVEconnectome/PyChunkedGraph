from flask import Blueprint
from middle_auth_client import auth_requires_permission

from pychunkedgraph.app.segmentation import common
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

bp = Blueprint("pcg_segmentation_v1", __name__, url_prefix="/segmentation/api/v1")


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


### GET ROOT -------------------------------------------------------------------


@bp.route("/table/<table_id>/sv/<atomic_id>/root", methods=["GET"])
def handle_root(table_id, atomic_id):
    return common.handle_root_2(table_id, atomic_id)


### MERGE ----------------------------------------------------------------------


@bp.route("/table/<table_id>/merge", methods=["POST"])
@auth_requires_permission("edit")
def handle_merge(table_id):
    return common.handle_merge(table_id)


### SPLIT ----------------------------------------------------------------------


@bp.route("/table/<table_id>/split", methods=["POST"])
@auth_requires_permission("edit")
def handle_split(table_id):
    return common.handle_split(table_id)


### UNDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/undo", methods=["POST"])
@auth_requires_permission("edit")
def handle_undo(table_id):
    return common.handle_undo(table_id)


### REDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/redo", methods=["POST"])
@auth_requires_permission("edit")
def handle_redo(table_id):
    return common.handle_redo(table_id)


### CHILDREN -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/children", methods=["GET"])
def handle_children(table_id, node_id):
    return common.handle_children(table_id, node_id)


### LEAVES ---------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/leaves", methods=["GET"])
def handle_leaves(table_id, node_id):
    return common.handle_leaves(table_id, node_id)


### SUBGRAPH -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/subgraph", methods=["GET"])
def handle_subgraph(table_id, node_id):
    return common.handle_subgraph(table_id, node_id)


### CHANGE LOG -----------------------------------------------------------------


@bp.route("/table/<table_id>/root/<root_id>/change_log", methods=["GET"])
def change_log(table_id, root_id):
    return common.change_log(table_id, root_id)


@bp.route("/table/<table_id>/root/<root_id>/merge_log", methods=["GET"])
def merge_log(table_id, root_id):
    return common.merge_log(table_id, root_id)


@bp.route("/table/<table_id>/oldest_timestamp", methods=["GET"])
def oldest_timestamp(table_id):
    return common.oldest_timestamp(table_id)


### CONTACT SITES --------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/contact_sites", methods=["GET"])
def handle_contact_sites(table_id, node_id):
    return common.handle_contact_sites(table_id, node_id)
