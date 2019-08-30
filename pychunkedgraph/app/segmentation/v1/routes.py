from flask import Blueprint, request
from middle_auth_client import auth_requires_permission

from pychunkedgraph.app.app_utils import jsonify_with_kwargs, toboolean
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


### MERGE ----------------------------------------------------------------------


@bp.route("/table/<table_id>/merge", methods=["POST"])
@auth_requires_permission("edit")
def handle_merge(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    merge_result = common.handle_merge(table_id)
    resp = {"operation_id": merge_result.operation_id, "new_root_ids": merge_result.new_root_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### SPLIT ----------------------------------------------------------------------


@bp.route("/table/<table_id>/split", methods=["POST"])
@auth_requires_permission("edit")
def handle_split(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    split_result = common.handle_split(table_id)
    resp = {"operation_id": split_result.operation_id, "new_root_ids": split_result.new_root_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### UNDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/undo", methods=["POST"])
@auth_requires_permission("edit")
def handle_undo(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    undo_result = common.handle_undo(table_id)
    resp = {"operation_id": undo_result.operation_id, "new_root_ids": undo_result.new_root_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### REDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/redo", methods=["POST"])
@auth_requires_permission("edit")
def handle_redo(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    redo_result = common.handle_redo(table_id)
    resp = {"operation_id": redo_result.operation_id, "new_root_ids": redo_result.new_root_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET ROOT -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/root", methods=["GET"])
@auth_requires_permission("view")
def handle_root(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    root_id = common.handle_root(table_id, node_id)
    return jsonify_with_kwargs({"root_id": root_id}, int64_as_str=int64_as_str)


### CHILDREN -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/children", methods=["GET"])
@auth_requires_permission("view")
def handle_children(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    children_ids = common.handle_children(table_id, node_id)
    resp = {"children_ids": children_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### LEAVES ---------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/leaves", methods=["GET"])
@auth_requires_permission("view")
def handle_leaves(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    leaf_ids = common.handle_leaves(table_id, node_id)
    resp = {"leaf_ids": leaf_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### SUBGRAPH -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/subgraph", methods=["GET"])
@auth_requires_permission("view")
def handle_subgraph(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    subgraph_result = common.handle_subgraph(table_id, node_id)
    resp = {"atomic_edges": subgraph_result}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### CONTACT SITES --------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/contact_sites", methods=["GET"])
@auth_requires_permission("view")
def handle_contact_sites(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    contact_sites = common.handle_contact_sites(table_id, node_id)
    return jsonify_with_kwargs(contact_sites, int64_as_str=int64_as_str)


### CHANGE LOG -----------------------------------------------------------------


@bp.route("/table/<table_id>/root/<root_id>/change_log", methods=["GET"])
@auth_requires_permission("view")
def change_log(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    log = common.change_log(table_id, root_id)
    return jsonify_with_kwargs(log, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root/<root_id>/merge_log", methods=["GET"])
@auth_requires_permission("view")
def merge_log(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    log = common.merge_log(table_id, root_id)
    return jsonify_with_kwargs(log, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/oldest_timestamp", methods=["GET"])
@auth_requires_permission("view")
def oldest_timestamp(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    delimiter = request.args.get("delimiter", default=" ", type=str)
    earliest_timestamp = common.oldest_timestamp(table_id)
    resp = {"iso": earliest_timestamp.isoformat(delimiter)}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root/<root_id>/last_edit", methods=["GET"])
@auth_requires_permission("view")
def last_edit(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    delimiter = request.args.get("delimiter", default=" ", type=str)
    latest_timestamp = common.last_edit(table_id, root_id)
    resp = {"iso": latest_timestamp.isoformat(delimiter)}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route('/table/<table_id>/graph/split_preview', methods=["POST"])
@auth_requires_permission("view")
def handle_split_preview(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    split_preview = common.handle_split_preview(table_id)
    return jsonify_with_kwargs(split_preview, int64_as_str=int64_as_str)
