import io
import csv

from flask import make_response, current_app
from flask import Blueprint, request
from middle_auth_client import auth_requires_permission
from middle_auth_client import auth_requires_admin
from middle_auth_client import auth_required

from pychunkedgraph.app.app_utils import jsonify_with_kwargs, toboolean, tobinary
from pychunkedgraph.app.segmentation import common
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

bp = Blueprint("pcg_segmentation_v1", __name__, url_prefix="/segmentation/api/v1")


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
    return common.before_request()


@bp.after_request
@auth_required
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


@bp.route('/table/<table_id>/graph/split_preview', methods=["POST"])
@auth_requires_permission("view")
def handle_split_preview(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    split_preview = common.handle_split_preview(table_id)
    return jsonify_with_kwargs(split_preview, int64_as_str=int64_as_str)


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
    resp = {"root_id": root_id}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET ROOTS ------------------------------------------------------------------


@bp.route("/table/<table_id>/roots", methods=["POST"])
@auth_requires_permission("view")
def handle_roots(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    root_ids = common.handle_roots(table_id, is_binary=False)
    resp = {"root_ids": root_ids}

    arg_as_binary = request.args.get("as_binary", default="", type=str)
    if arg_as_binary in resp:
        return tobinary(resp[arg_as_binary])
    else:
        return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)

### GET ROOTS BINARY -----------------------------------------------------------

@bp.route("/table/<table_id>/roots_binary", methods=["POST"])
@auth_requires_permission("view")
def handle_roots_binary(table_id):
    root_ids = common.handle_roots(table_id, is_binary=True)
    return tobinary(root_ids)


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
    contact_sites, contact_site_metadata = common.handle_contact_sites(
        table_id, node_id
    )
    resp = {
        "contact_sites": contact_sites,
        "contact_site_metadata": contact_site_metadata,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/node/contact_sites_pair/<first_node_id>/<second_node_id>", methods=["GET"])
@auth_requires_permission("view")
def handle_pairwise_contact_sites(table_id, first_node_id, second_node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    contact_sites, contact_site_metadata = common.handle_pairwise_contact_sites(
        table_id, first_node_id, second_node_id
    )
    resp = {
        "contact_sites": contact_sites,
        "contact_site_metadata": contact_site_metadata,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)

### CHANGE LOG -----------------------------------------------------------------


@bp.route("/table/<table_id>/change_log", methods=["GET"])
@auth_requires_admin
def change_log_full(table_id):
    si = io.StringIO()
    cw = csv.writer(si)
    log_entries = common.change_log(table_id)
    cw.writerow(["user_id","action","root_ids","timestamp"])
    cw.writerows(log_entries)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={table_id}.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@bp.route("/table/<table_id>/tabular_change_log_weekly", methods=["GET"])
@auth_requires_permission("view")
def tabular_change_log_weekly(table_id):
    disp = request.args.get("disp", default=False, type=toboolean)
    weekly_tab_change_log = common.tabular_change_log_weekly(table_id)

    if disp:
        return weekly_tab_change_log.to_html()
    else:
        return weekly_tab_change_log.to_json()


@bp.route("/table/<table_id>/root/<root_id>/change_log", methods=["GET"])
@auth_requires_permission("view")
def change_log(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    log = common.change_log(table_id, root_id)
    return jsonify_with_kwargs(log, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root/<root_id>/tabular_change_log", methods=["GET"])
@auth_requires_permission("view")
def tabular_change_log(table_id, root_id):
    disp = request.args.get("disp", default=False, type=toboolean)
    tab_change_log = common.tabular_change_log(table_id, root_id)

    if disp:
        return tab_change_log.to_html()
    else:
        return tab_change_log.to_json()

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


### FIND PATH ------------------------------------------------------------------


@bp.route("/table/<table_id>/graph/find_path", methods=["POST"])
@auth_requires_permission("view")
def find_path(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    find_path_result = common.handle_find_path(table_id)
    return jsonify_with_kwargs(find_path_result, int64_as_str=int64_as_str)
