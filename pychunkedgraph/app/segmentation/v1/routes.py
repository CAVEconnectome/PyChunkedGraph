from flask import Blueprint, jsonify, request
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
@auth_requires_permission("view")
def handle_root(table_id, atomic_id):
    root_id = common.handle_root(table_id, atomic_id)
    return jsonify({"root_id": root_id})


### MERGE ----------------------------------------------------------------------


@bp.route("/table/<table_id>/merge", methods=["POST"])
@auth_requires_permission("edit")
def handle_merge(table_id):
    merge_result = common.handle_merge(table_id)
    resp = {
        "operation_id": merge_result.operation_id,
        "new_root_ids": merge_result.new_root_ids,
    }
    return jsonify(resp)


### SPLIT ----------------------------------------------------------------------


@bp.route("/table/<table_id>/split", methods=["POST"])
@auth_requires_permission("edit")
def handle_split(table_id):
    split_result = common.handle_split(table_id)
    resp = {
        "operation_id": split_result.operation_id,
        "new_root_ids": split_result.new_root_ids,
    }
    return jsonify(resp)


### UNDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/undo", methods=["POST"])
@auth_requires_permission("edit")
def handle_undo(table_id):
    undo_result = common.handle_undo(table_id)
    resp = {
        "operation_id": undo_result.operation_id,
        "new_root_ids": undo_result.new_root_ids,
    }
    return jsonify(resp)


### REDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/redo", methods=["POST"])
@auth_requires_permission("edit")
def handle_redo(table_id):
    redo_result = common.handle_redo(table_id)
    resp = {
        "operation_id": redo_result.operation_id,
        "new_root_ids": redo_result.new_root_ids,
    }
    return jsonify(resp)


### CHILDREN -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/children", methods=["GET"])
@auth_requires_permission("view")
def handle_children(table_id, node_id):
    children_ids = common.handle_children(table_id, node_id)
    resp = {"children_ids": children_ids}
    return jsonify(resp)


### LEAVES ---------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/leaves", methods=["GET"])
@auth_requires_permission("view")
def handle_leaves(table_id, node_id):
    leaf_ids = common.handle_leaves(table_id, node_id)
    resp = {"leaf_ids": leaf_ids}
    return jsonify(resp)


### SUBGRAPH -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/subgraph", methods=["GET"])
@auth_requires_permission("view")
def handle_subgraph(table_id, node_id):
    subgraph_result = common.handle_subgraph(table_id, node_id)
    resp = {"atomic_edges": subgraph_result}
    return jsonify(resp)


### CHANGE LOG -----------------------------------------------------------------


@bp.route("/table/<table_id>/root/<root_id>/change_log", methods=["GET"])
@auth_requires_permission("view")
def change_log(table_id, root_id):
    log = common.change_log(table_id, root_id)
    return jsonify(log)


@bp.route("/table/<table_id>/root/<root_id>/merge_log", methods=["GET"])
@auth_requires_permission("view")
def merge_log(table_id, root_id):
    log = common.merge_log(table_id, root_id)
    return jsonify(log)


@bp.route("/table/<table_id>/oldest_timestamp", methods=["GET"])
@auth_requires_permission("view")
def oldest_timestamp(table_id):
    earliest_timestamp = common.oldest_timestamp(table_id)
    resp = {"iso": str(earliest_timestamp)}
    return jsonify(resp)


@bp.route("/table/<table_id>/root/<root_id>/last_edit", methods=["GET"])
@auth_requires_permission("view")
def last_edit(table_id, root_id):
    latest_timestamp = common.last_edit(table_id, root_id)
    resp = {"iso": str(latest_timestamp)}
    return jsonify(resp)


### CONTACT SITES --------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/contact_sites", methods=["GET"])
@auth_requires_permission("view")
def handle_contact_sites(table_id, node_id):
    contact_sites = common.handle_contact_sites(table_id, node_id)
    return jsonify(contact_sites)
