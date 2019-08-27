from flask import Blueprint
from middle_auth_client import auth_requires_permission

from pychunkedgraph.app.segmentation import common
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

bp = Blueprint("pcg_segmentation_v0", __name__, url_prefix="/segmentation/")


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


# -------------------
# ------ Applications
# -------------------


@bp.route("/sleep/<int:sleep>")
def sleep_me(sleep):
    return common.sleep_me(sleep)


@bp.route("/<api_version>/<table_id>/info", methods=["GET"])
def handle_info(table_id):
    return common.handle_info(table_id)


### GET ROOT -------------------------------------------------------------------


@bp.route("/1.0/<table_id>/graph/root", methods=["POST", "GET"])
def handle_root_1(table_id):
    return common.handle_root_1(table_id)


@bp.route("/1.0/<table_id>/graph/<atomic_id>/root", methods=["POST", "GET"])
def handle_root_2(table_id, atomic_id):
    return common.handle_root_2(table_id, atomic_id)


### MERGE ----------------------------------------------------------------------


@bp.route("/1.0/<table_id>/graph/merge", methods=["POST", "GET"])
@auth_requires_permission("edit")
def handle_merge(table_id):
    return common.handle_merge(table_id)


### SPLIT ----------------------------------------------------------------------


@bp.route("/1.0/<table_id>/graph/split", methods=["POST", "GET"])
@auth_requires_permission("edit")
def handle_split(table_id):
    return common.handle_split(table_id)


### UNDO ----------------------------------------------------------------------


@bp.route("/1.0/<table_id>/graph/undo", methods=["POST"])
@auth_requires_permission("edit")
def handle_undo(table_id):
    return common.handle_undo(table_id)


### REDO ----------------------------------------------------------------------


@bp.route("/1.0/<table_id>/graph/redo", methods=["POST"])
@auth_requires_permission("edit")
def handle_redo(table_id):
    return common.handle_redo(table_id)


### CHILDREN -------------------------------------------------------------------


@bp.route("/1.0/<table_id>/segment/<parent_id>/children", methods=["POST", "GET"])
def handle_children(table_id, parent_id):
    return common.handle_children(table_id, parent_id)


### LEAVES ---------------------------------------------------------------------


@bp.route("/1.0/<table_id>/segment/<root_id>/leaves", methods=["POST", "GET"])
def handle_leaves(table_id, root_id):
    return common.handle_leaves(table_id, root_id)


### LEAVES FROM LEAVES ---------------------------------------------------------


@bp.route(
    "/1.0/<table_id>/segment/<atomic_id>/leaves_from_leave", methods=["POST", "GET"]
)
def handle_leaves_from_leave(table_id, atomic_id):
    return common.handle_leaves_from_leave(table_id, atomic_id)


### SUBGRAPH -------------------------------------------------------------------


@bp.route("/1.0/<table_id>/segment/<root_id>/subgraph", methods=["POST", "GET"])
def handle_subgraph(table_id, root_id):
    return common.handle_subgraph(table_id, root_id)


### CHANGE LOG -----------------------------------------------------------------


@bp.route("/1.0/<table_id>/segment/<root_id>/change_log", methods=["POST", "GET"])
def change_log(table_id, root_id):
    return common.change_log(table_id, root_id)


@bp.route("/1.0/<table_id>/segment/<root_id>/merge_log", methods=["POST", "GET"])
def merge_log(table_id, root_id):
    return common.merge_log(table_id, root_id)


@bp.route("/1.0/<table_id>/graph/oldest_timestamp", methods=["POST", "GET"])
def oldest_timestamp(table_id):
    return common.oldest_timestamp(table_id)


### CONTACT SITES --------------------------------------------------------------


@bp.route("/1.0/<table_id>/segment/<root_id>/contact_sites", methods=["POST", "GET"])
def handle_contact_sites(table_id, root_id):
    return common.handle_contact_sites(table_id, root_id)
