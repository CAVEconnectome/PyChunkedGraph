import json

import numpy as np

from flask import Blueprint, jsonify, request
from middle_auth_client import auth_requires_permission
from pychunkedgraph.app import app_utils
from pychunkedgraph.app.segmentation import common
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

bp = Blueprint("pcg_segmentation_v0", __name__, url_prefix="/segmentation/1.0")


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


@bp.route("/<table_id>/info", methods=["GET"])
@auth_requires_permission("view")
def handle_info(table_id):
    return common.handle_info(table_id)


### MERGE ----------------------------------------------------------------------


@bp.route("/<table_id>/graph/merge", methods=["POST", "GET"])
@auth_requires_permission("edit")
def handle_merge(table_id):
    merge_result = common.handle_merge(table_id)
    return app_utils.tobinary(merge_result.new_root_ids)


### SPLIT ----------------------------------------------------------------------


@bp.route("/<table_id>/graph/split", methods=["POST", "GET"])
@auth_requires_permission("edit")
def handle_split(table_id):
    split_result = common.handle_split(table_id)
    return app_utils.tobinary(split_result.new_root_ids)


### GET ROOT -------------------------------------------------------------------


@bp.route("/<table_id>/graph/root", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_root_1(table_id):
    atomic_id = np.uint64(json.loads(request.data)[0])
    root_id = common.handle_root(table_id, atomic_id)
    return app_utils.tobinary(root_id)


@bp.route("/<table_id>/graph/<atomic_id>/root", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_root_2(table_id, atomic_id):
    root_id = common.handle_root(table_id, atomic_id)
    return app_utils.tobinary(root_id)


### CHILDREN -------------------------------------------------------------------


@bp.route("/<table_id>/segment/<parent_id>/children", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_children(table_id, parent_id):
    children_ids = common.handle_children(table_id, parent_id)
    return app_utils.tobinary(children_ids)


### LEAVES ---------------------------------------------------------------------


@bp.route("/<table_id>/segment/<root_id>/leaves", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_leaves(table_id, root_id):
    leaf_ids = common.handle_leaves(table_id, root_id)
    return app_utils.tobinary(leaf_ids)


### LEAVES FROM LEAVES ---------------------------------------------------------


@bp.route("/<table_id>/segment/<atomic_id>/leaves_from_leave", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_leaves_from_leave(table_id, atomic_id):
    leaf_ids = common.handle_leaves_from_leave(table_id, atomic_id)
    return app_utils.tobinary(leaf_ids)


### SUBGRAPH -------------------------------------------------------------------


@bp.route("/<table_id>/segment/<root_id>/subgraph", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_subgraph(table_id, root_id):
    subgraph_result = common.handle_subgraph(table_id, root_id)
    return app_utils.tobinary(subgraph_result)


### CONTACT SITES --------------------------------------------------------------


@bp.route("/<table_id>/segment/<root_id>/contact_sites", methods=["POST", "GET"])
@auth_requires_permission("view")
def handle_contact_sites(table_id, root_id):
    contact_sites = common.handle_contact_sites(table_id, root_id)
    return jsonify(contact_sites)


### CHANGE LOG -----------------------------------------------------------------


@bp.route("/<table_id>/segment/<root_id>/change_log", methods=["POST", "GET"])
@auth_requires_permission("view")
def change_log(table_id, root_id):
    log = common.change_log(table_id, root_id)
    return jsonify(log)


@bp.route("/<table_id>/segment/<root_id>/merge_log", methods=["POST", "GET"])
@auth_requires_permission("view")
def merge_log(table_id, root_id):
    log = common.merge_log(table_id, root_id)
    return jsonify(log)


@bp.route("/<table_id>/graph/oldest_timestamp", methods=["POST", "GET"])
@auth_requires_permission("view")
def oldest_timestamp(table_id):
    earliest_timestamp = common.oldest_timestamp(table_id)
    resp = {"iso": str(earliest_timestamp)}
    return jsonify(resp)
