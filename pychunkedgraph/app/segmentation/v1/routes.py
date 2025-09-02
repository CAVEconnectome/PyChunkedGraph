# pylint: disable=invalid-name, missing-docstring

import csv
import io
import json
import pickle

import numpy as np
import pandas as pd
from flask import Blueprint, make_response, request
from middle_auth_client import (
    auth_required,
    auth_requires_admin,
    auth_requires_permission,
)

from pychunkedgraph.app import common as app_common
from pychunkedgraph.app import app_utils
from pychunkedgraph.app.app_utils import (
    jsonify_with_kwargs,
    remap_public,
    tobinary,
    toboolean,
)
from pychunkedgraph.app.segmentation import common
from pychunkedgraph.graph import exceptions as cg_exceptions

bp = Blueprint(
    "pcg_segmentation_v1",
    __name__,
    url_prefix=f"/{common.__segmentation_url_prefix__}/api/v1",
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


### MERGE ----------------------------------------------------------------------


@bp.route("/table/<table_id>/merge", methods=["POST"])
@auth_requires_permission("edit")
@remap_public(edit=True)
def handle_merge(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    merge_result = common.handle_merge(table_id)
    resp = {
        "operation_id": merge_result.operation_id,
        "new_root_ids": merge_result.new_root_ids,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/merge_admin", methods=["POST"])
@auth_requires_permission("admin")
@remap_public(edit=True)
def handle_merge_admin(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    allow_same_segment_merge = request.args.get(
        "allow_same_segment_merge", False, type=common.str2bool
    )
    merge_result = common.handle_merge(
        table_id, allow_same_segment_merge=allow_same_segment_merge
    )
    resp = {
        "operation_id": merge_result.operation_id,
        "new_root_ids": merge_result.new_root_ids,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### SPLIT ----------------------------------------------------------------------


@bp.route("/table/<table_id>/split", methods=["POST"])
@auth_requires_permission("edit")
@remap_public(edit=True)
def handle_split(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    split_result = common.handle_split(table_id)
    resp = {
        "operation_id": split_result.operation_id,
        "new_root_ids": split_result.new_root_ids,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/graph/split_preview", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=True)
def handle_split_preview(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    split_preview = common.handle_split_preview(table_id)
    return jsonify_with_kwargs(split_preview, int64_as_str=int64_as_str)


### UNDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/undo", methods=["POST"])
@auth_requires_permission("edit")
@remap_public(edit=True)
def handle_undo(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    undo_result = common.handle_undo(table_id)
    resp = {
        "operation_id": undo_result.operation_id,
        "new_root_ids": undo_result.new_root_ids,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### REDO ----------------------------------------------------------------------


@bp.route("/table/<table_id>/redo", methods=["POST"])
@auth_requires_permission("edit")
@remap_public(edit=True)
def handle_redo(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    redo_result = common.handle_redo(table_id)
    resp = {
        "operation_id": redo_result.operation_id,
        "new_root_ids": redo_result.new_root_ids,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### ROLLBACK USER --------------------------------------------------------------


@bp.route("/table/<table_id>/rollback_user", methods=["POST"])
@auth_requires_admin
@remap_public(edit=True)
def handle_rollback(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    rollback_result = common.handle_rollback(table_id)
    resp = rollback_result
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### USER OPERATIONS -------------------------------------------------------------


@bp.route("/table/<table_id>/user_operations", methods=["GET"])
@auth_requires_permission("admin_view")
@remap_public(edit=True)
def handle_user_operations(table_id):
    disp = request.args.get("disp", default=False, type=toboolean)
    include_undone = request.args.get("include_undone", default=False, type=toboolean)
    user_operations = pd.DataFrame.from_dict(
        common.all_user_operations(table_id, include_undone)
    )

    if disp:
        return user_operations.to_html()
    else:
        return user_operations.to_json()


### GET ROOT -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/root", methods=["GET"])
@auth_requires_permission("view")
@remap_public
def handle_root(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    root_id = common.handle_root(table_id, node_id)
    resp = {"root_id": root_id}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET ROOTS ------------------------------------------------------------------


@bp.route("/table/<table_id>/roots", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=False)
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
@remap_public(edit=False)
def handle_roots_binary(table_id):
    root_ids = common.handle_roots(table_id, is_binary=True)
    return tobinary(root_ids)


### CHILDREN -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/children", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_children(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    children_ids = common.handle_children(table_id, node_id)
    resp = {"children_ids": children_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET L2:SV MAPPINGS OF A L2 CHUNK ------------------------------------------------------------------


@bp.route("/table/<table_id>/l2_chunk_children/<chunk_id>", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_l2_chunk_children(table_id, chunk_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    as_array = request.args.get("as_array", default=False, type=toboolean)
    l2_chunk_children = common.handle_l2_chunk_children(table_id, chunk_id, as_array)
    if as_array:
        resp = {"l2_chunk_children": l2_chunk_children}
    else:
        resp = {"l2_chunk_children": pickle.dumps(l2_chunk_children)}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET L2:SV MAPPINGS OF A L2 CHUNK BINARY ------------------------------------------------------------------


@bp.route("/table/<table_id>/l2_chunk_children_binary/<chunk_id>", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_l2_chunk_children_binary(table_id, chunk_id):
    as_array = request.args.get("as_array", default=False, type=toboolean)
    l2_chunk_children = common.handle_l2_chunk_children(table_id, chunk_id, as_array)
    if as_array:
        return tobinary(l2_chunk_children)
    else:
        return pickle.dumps(l2_chunk_children)


### LEAVES ---------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/leaves", methods=["GET"])
@auth_requires_permission(
    "view",
    public_table_key="table_id",
    public_node_key="node_id",
)
@remap_public(edit=False)
def handle_leaves(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    leaf_ids = common.handle_leaves(table_id, node_id)
    resp = {"leaf_ids": leaf_ids}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


## LEAVES OF MANY ROOTS


@bp.route("/table/<table_id>/node/leaves_many", methods=["POST"])
@bp.route("/table/<table_id>/leaves_many", methods=["POST"])
@auth_requires_permission(
    "view",
    public_table_key="table_id",
    public_node_key="node_id",
)
@remap_public(check_node_ids=True)
def handle_leaves_many(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    root_to_leaf_dict = common.handle_leaves_many(table_id)
    return jsonify_with_kwargs(root_to_leaf_dict, int64_as_str=int64_as_str)


### GET MINIMAL COVERING NODES


@bp.route("/table/<table_id>/minimal_covering_nodes", methods=["POST"])
@auth_requires_permission("view", public_table_key="table_id")
@remap_public(check_node_ids=False)
def handle_minimal_covering_nodes(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    is_binary = request.args.get("is_binary", default=False, type=toboolean)

    covering_nodes = common.handle_find_minimal_covering_nodes(
        table_id, is_binary=is_binary
    )
    as_array = request.args.get("as_array", default=False, type=toboolean)
    if as_array:
        return tobinary(covering_nodes)
    return jsonify_with_kwargs(covering_nodes, int64_as_str=int64_as_str)


### SUBGRAPH -------------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/subgraph", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_subgraph(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    only_internal_edges = request.args.get(
        "only_internal_edges", default=True, type=toboolean
    )
    edges = common.handle_subgraph(
        table_id, node_id, only_internal_edges=only_internal_edges
    )
    resp = {
        "nodes": edges.get_pairs(),
        "affinities": edges.affinities,
        "areas": edges.areas,
    }
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### CONTACT SITES --------------------------------------------------------------


@bp.route("/table/<table_id>/node/<node_id>/contact_sites", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
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


@bp.route(
    "/table/<table_id>/node/contact_sites_pair/<first_node_id>/<second_node_id>",
    methods=["GET"],
)
@auth_requires_permission("view")
@remap_public(edit=False)
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
@remap_public(edit=False)
def change_log_full(table_id):
    si = io.StringIO()
    cw = csv.writer(si)
    log_entries = common.change_log(table_id)
    cw.writerow(["user_id", "action", "root_ids", "timestamp"])
    cw.writerows(log_entries)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={table_id}.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@bp.route("/table/<table_id>/tabular_change_log_recent", methods=["GET"])
@auth_requires_permission("admin_view")
@remap_public(edit=True)
def tabular_change_log_weekly(table_id):
    disp = request.args.get("disp", default=False, type=toboolean)
    weekly_tab_change_log = common.tabular_change_log_recent(table_id)

    if disp:
        return weekly_tab_change_log.to_html()
    else:
        return weekly_tab_change_log.to_json()


@bp.route("/table/<table_id>/root/<root_id>/change_log", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def change_log(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    filtered = request.args.get("filtered", default=False, type=toboolean)
    log = common.change_log(table_id, root_id, filtered)
    return jsonify_with_kwargs(log, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root/<root_id>/tabular_change_log", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def tabular_change_log(table_id, root_id):
    disp = request.args.get("disp", default=False, type=toboolean)
    # get_root_ids = request.args.get("root_ids", default=False, type=toboolean)
    filtered = request.args.get("filtered", default=True, type=toboolean)
    tab_change_log_dict = common.tabular_change_logs(table_id, [int(root_id)], filtered)
    tab_change_log = tab_change_log_dict[int(root_id)]
    if disp:
        return tab_change_log.to_html()
    return tab_change_log.to_json()


@bp.route("/table/<table_id>/tabular_change_log_many", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def tabular_change_log_many(table_id):
    filtered = request.args.get("filtered", default=True, type=toboolean)
    root_ids = np.array(json.loads(request.data)["root_ids"], dtype=np.uint64)
    tab_change_log_dict = common.tabular_change_logs(table_id, root_ids, filtered)

    return jsonify_with_kwargs(
        {str(k): tab_change_log_dict[k] for k in tab_change_log_dict.keys()}
    )


@bp.route("/table/<table_id>/root/<root_id>/merge_log", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def merge_log(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    log = common.merge_log(table_id, root_id)
    return jsonify_with_kwargs(log, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root/<root_id>/lineage_graph", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_lineage_graph(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.handle_lineage_graph(table_id, root_id)
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/lineage_graph_multiple", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_lineage_graph_multiple(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.handle_lineage_graph(table_id)
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/past_id_mapping", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_past_id_mapping(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.handle_past_id_mapping(table_id)
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/oldest_timestamp", methods=["GET"])
@auth_requires_permission(
    "view",
    public_table_key="table_id",
)
@remap_public(edit=False)
def oldest_timestamp(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    delimiter = request.args.get("delimiter", default=" ", type=str)
    earliest_timestamp = common.oldest_timestamp(table_id)
    resp = {"iso": earliest_timestamp.isoformat(delimiter)}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root/<root_id>/last_edit", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def last_edit(table_id, root_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    delimiter = request.args.get("delimiter", default=" ", type=str)
    latest_timestamp = common.last_edit(table_id, root_id)
    resp = {"iso": latest_timestamp.isoformat(delimiter)}
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### FIND PATH ------------------------------------------------------------------


@bp.route("/table/<table_id>/graph/find_path", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=False)
def find_path(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    precision_mode = request.args.get("precision_mode", default=True, type=toboolean)
    find_path_result = common.handle_find_path(table_id, precision_mode)
    return jsonify_with_kwargs(find_path_result, int64_as_str=int64_as_str)


### ROOT INFO -----------------------------------------------------------------


@bp.route("/table/<table_id>/is_latest_roots", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_is_latest_roots(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    is_latest_roots = common.handle_is_latest_roots(table_id, is_binary=False)
    resp = {"is_latest": is_latest_roots}

    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/root_timestamps", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_root_timestamps(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    is_binary = request.args.get("is_binary", default=False, type=toboolean)
    latest = request.args.get("latest", default=False, type=toboolean)
    is_binary = request.args.get("is_binary", default=False, type=toboolean)
    root_timestamps = common.handle_root_timestamps(
        table_id, is_binary=is_binary, latest=latest
    )
    resp = {"timestamp": root_timestamps}

    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


## Lookup root id from coordinate -----------------------------------------------


@bp.route("/table/<table_id>/roots_from_coords", methods=["POST"])
@auth_requires_permission("view")
@remap_public(edit=False)
def handle_roots_from_coords(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.handle_roots_from_coord(table_id)
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


## Get level2 graph -------------------------------------------------------------
@bp.route("/table/<table_id>/node/<node_id>/lvl2_graph", methods=["GET"])
@auth_requires_permission(
    "view",
    public_table_key="table_id",
    public_node_key="node_id",
)
@remap_public(edit=False)
def handle_get_lvl2_graph(table_id, node_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.handle_get_layer2_graph(table_id, node_id)
    out = jsonify_with_kwargs(resp, int64_as_str=int64_as_str)
    if "bounds" in request.args:
        out.headers["Used-Bounds"] = True
    else:
        out.headers["Used-Bounds"] = False
    return out


### GET OPERATION DETAILS --------------------------------------------------------


@bp.route("/table/<table_id>/operation_details", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def operation_details(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.operation_details(table_id)
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET PROOFREAD IDS --------------------------------------------------------


@bp.route("/table/<table_id>/delta_roots", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def delta_roots(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    resp = common.delta_roots(table_id)
    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


### GET VALID NODES -------------------------------------------------------------


@bp.route("/table/<table_id>/valid_nodes", methods=["GET"])
@auth_requires_permission("view")
@remap_public(edit=False)
def valid_nodes(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)
    is_binary = request.args.get("is_binary", default=False, type=toboolean)
    resp = common.valid_nodes(table_id, is_binary=is_binary)

    return jsonify_with_kwargs(resp, int64_as_str=int64_as_str)


@bp.route("/table/<table_id>/supervoxel_lookup", methods=["POST"])
@auth_requires_permission("admin")
@remap_public(edit=False)
def handle_supervoxel_lookup(table_id):
    int64_as_str = request.args.get("int64_as_str", default=False, type=toboolean)

    nodes = json.loads(request.data)
    cg = app_utils.get_cg(table_id)
    node_ids = []
    coords = []
    for node in nodes:
        node_ids.append(node[0])
        coords.append(np.array(node[1:]) / cg.segmentation_resolution)

    atomic_ids = app_utils.handle_supervoxel_id_lookup(cg, coords, node_ids)
    return jsonify_with_kwargs(atomic_ids, int64_as_str=int64_as_str)
