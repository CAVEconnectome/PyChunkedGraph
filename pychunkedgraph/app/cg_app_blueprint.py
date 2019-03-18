from flask import Blueprint, request, make_response, jsonify, current_app,\
    redirect, url_for

import json
import numpy as np
import time
from datetime import datetime
from pytz import UTC
import traceback
import collections

from pychunkedgraph.app import app_utils
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.backend import chunkedgraph_utils

__version__ = '0.1.104'
bp = Blueprint('pychunkedgraph', __name__, url_prefix="/segmentation")

# -------------------------------
# ------ Access control and index
# -------------------------------


@bp.route('/')
@bp.route("/index")
def index():
    return "PyChunkedGraph Server -- " + __version__


@bp.route
def home():
    resp = make_response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


# -------------------------------
# ------ Measurements and Logging
# -------------------------------

@bp.before_request
def before_request():
    current_app.request_start_time = time.time()
    current_app.request_start_date = datetime.utcnow()


@bp.after_request
def after_request(response):
    dt = (time.time() - current_app.request_start_time) * 1000

    current_app.logger.debug("Response time: %.3fms" % dt)
    return response


@bp.errorhandler(Exception)
def unhandled_exception(e):
    status_code = 500
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e,
                                    tb=e.__traceback__)

    current_app.logger.error({
        "message": str(e),
        "user_id": user_ip,
        "user_ip": user_ip,
        "request_time": current_app.request_start_date,
        "request_url": request.url,
        "request_data": request.data,
        "response_time": response_time,
        "response_code": status_code,
        "traceback": tb
    })

    resp = {
        'timestamp': current_app.request_start_date,
        'duration': response_time,
        'code': status_code,
        'message': str(e),
        'traceback': tb
    }

    return jsonify(resp), status_code


@bp.errorhandler(cg_exceptions.ChunkedGraphAPIError)
def api_exception(e):
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e,
                                    tb=e.__traceback__)

    current_app.logger.error({
        "message": str(e),
        "user_id": user_ip,
        "user_ip": user_ip,
        "request_time": current_app.request_start_date,
        "request_url": request.url,
        "request_data": request.data,
        "response_time": response_time,
        "response_code": e.status_code.value,
        "traceback": tb
    })

    resp = {
        'timestamp': current_app.request_start_date,
        'duration': response_time,
        'code': e.status_code.value,
        'message': str(e)
    }

    return jsonify(resp), e.status_code.value


# -------------------
# ------ Applications
# -------------------


@bp.route("/sleep/<int:sleep>")
def sleep_me(sleep):
    time.sleep(sleep)
    return "zzz... {} ... awake".format(sleep)


@bp.route('/1.0/<table_id>/info', methods=['GET'])
def handle_info(table_id):
    cg = app_utils.get_cg(table_id)

    return json.dumps(obj=app_utils.make_json_serializable(cg.dataset_info),
                      default=app_utils.json_defaults)


### GET ROOT -------------------------------------------------------------------

@bp.route('/1.0/graph/root', methods=['POST', 'GET'])
def handle_root_1():
    atomic_id = np.uint64(json.loads(request.data)[0])
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']

    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get('timestamp', time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError) as e:
        raise(cg_exceptions.BadRequest("Timestamp parameter is not a valid unix timestamp"))

    return handle_root_main(table_id, np.uint64(atomic_id), timestamp)


@bp.route('/1.0/<table_id>/graph/root', methods=['POST', 'GET'])
def handle_root_2(table_id):
    atomic_id = np.uint64(json.loads(request.data)[0])

    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get('timestamp', time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError) as e:
        raise(cg_exceptions.BadRequest("Timestamp parameter is not a valid unix timestamp"))

    return handle_root_main(table_id, atomic_id, timestamp)


@bp.route('/1.0/<table_id>/graph/<atomic_id>/root', methods=['POST', 'GET'])
def handle_root_3(table_id, atomic_id):

    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get('timestamp', time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError) as e:
        raise(cg_exceptions.BadRequest("Timestamp parameter is not a valid unix timestamp"))

    return handle_root_main(table_id, np.uint64(atomic_id), timestamp)


def handle_root_main(table_id, atomic_id, timestamp):

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    root_id = cg.get_root(atomic_id, time_stamp=timestamp)

    # Return binary
    return app_utils.tobinary(root_id)


### MERGE ----------------------------------------------------------------------

@bp.route('/1.0/graph/merge', methods=['POST', 'GET'])
def handle_merge_1():
    nodes = json.loads(request.data)
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
    user_id = str(request.remote_addr)

    return handle_merge_main(table_id, nodes, user_id)


@bp.route('/1.0/<table_id>/graph/merge', methods=['POST', 'GET'])
def handle_merge_2(table_id):
    nodes = json.loads(request.data)
    user_id = str(request.remote_addr)

    return handle_merge_main(table_id, nodes, user_id)


def handle_merge_main(table_id, nodes, user_id):
    current_app.logger.debug(nodes)
    assert len(nodes) == 2

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    atomic_edge = []
    coords = []
    for node in nodes:
        node_id = node[0]
        x, y, z = node[1:]

        x /= 2
        y /= 2

        coordinate = np.array([x, y, z])

        if not cg.is_in_bounds(coordinate):
            coordinate /= cg.segmentation_resolution

            coordinate[0] *= 2
            coordinate[1] *= 2

        atomic_id = cg.get_atomic_id_from_coord(coordinate[0],
                                                coordinate[1],
                                                coordinate[2],
                                                parent_id=np.uint64(node_id))

        if atomic_id is None:
            raise cg_exceptions.BadRequest(
                f"Could not determine supervoxel ID for coordinates {coordinate}."
            )

        coords.append(coordinate)
        atomic_edge.append(atomic_id)

    # Protection from long range mergers
    chunk_coord_delta = cg.get_chunk_coordinates(atomic_edge[0]) - \
                        cg.get_chunk_coordinates(atomic_edge[1])

    if np.any(np.abs(chunk_coord_delta) > 3):
        raise cg_exceptions.BadRequest(
            "Chebyshev distance between merge points exceeded allowed maximum (3 chunks).")

    try:
        new_root = cg.add_edges(user_id=user_id,
                                atomic_edges=np.array(atomic_edge, dtype=np.uint64),
                                source_coord=coords[:1],
                                sink_coord=coords[1:],
                                remesh_preview=True)
    except cg_exceptions.LockingError as e:
        raise cg_exceptions.InternalServerError(
            "Could not acquire root lock for merge operation.")
    except cg_exceptions.PreconditionError as e:
        raise cg_exceptions.BadRequest(str(e))

    if new_root is None:
        raise cg_exceptions.InternalServerError(
            "Could not merge selected supervoxel.")

    # Return binary
    return app_utils.tobinary(new_root)


### SPLIT ----------------------------------------------------------------------


@bp.route('/1.0/graph/split', methods=['POST', 'GET'])
def handle_split_1():
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
    data = json.loads(request.data)
    user_id = str(request.remote_addr)

    return handle_split_main(table_id, data, user_id)


@bp.route('/1.0/<table_id>/graph/split', methods=['POST', 'GET'])
def handle_split_2(table_id):
    data = json.loads(request.data)
    user_id = str(request.remote_addr)

    return handle_split_main(table_id, data, user_id)


def handle_split_main(table_id, data, user_id):
    current_app.logger.debug(data)

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    data_dict = {}
    for k in ["sources", "sinks"]:
        data_dict[k] = collections.defaultdict(list)

        for node in data[k]:
            node_id = node[0]
            x, y, z = node[1:]

            x /= 2
            y /= 2

            coordinate = np.array([x, y, z])

            current_app.logger.debug(("before", coordinate))

            if not cg.is_in_bounds(coordinate):
                coordinate /= cg.segmentation_resolution

                coordinate[0] *= 2
                coordinate[1] *= 2

            current_app.logger.debug(("after", coordinate))

            atomic_id = cg.get_atomic_id_from_coord(coordinate[0],
                                                    coordinate[1],
                                                    coordinate[2],
                                                    parent_id=np.uint64(
                                                        node_id))

            if atomic_id is None:
                raise cg_exceptions.BadRequest(
                    f"Could not determine supervoxel ID for coordinates {coordinate}.")

            data_dict[k]["id"].append(atomic_id)
            data_dict[k]["coord"].append(coordinate)

    current_app.logger.debug(data_dict)

    try:
        new_roots = cg.remove_edges(user_id=user_id,
                                    source_ids=data_dict["sources"]["id"],
                                    sink_ids=data_dict["sinks"]["id"],
                                    source_coords=data_dict["sources"]["coord"],
                                    sink_coords=data_dict["sinks"]["coord"],
                                    mincut=True,
                                    remesh_preview=True)
    except cg_exceptions.LockingError as e:
        raise cg_exceptions.InternalServerError(
            "Could not acquire root lock for split operation.")
    except cg_exceptions.PreconditionError as e:
        raise cg_exceptions.BadRequest(str(e))

    if new_roots is None:
        raise cg_exceptions.InternalServerError(
            "Could not split selected segment groups."
        )

    current_app.logger.debug(("after split:", new_roots))

    # Return binary
    return app_utils.tobinary(new_roots)


### SHATTER --------------------------------------------------------------------

# @bp.route('/1.0/graph/shatter', methods=['POST', 'GET'])
# def handle_shatter_re():
#     return redirect('/1.0/%s/graph/shatter' %
#                     current_app.config['CHUNKGRAPH_TABLE_ID'])
#
#
# @bp.route('/1.0/<table_id>/graph/shatter', methods=['POST', 'GET'])
# def handle_shatter(table_id):
#     data = json.loads(request.data)
#
#     user_id = str(request.remote_addr)
#
#     current_app.logger.debug(data)
#
#     # Call ChunkedGraph
#     cg = app_utils.get_cg(table_id)
#
#     data_dict = collections.defaultdict(list)
#
#     k = "sources"
#     node = data[k][0]
#
#     node_id = node[0]
#     radius = node[1]
#     x, y, z = node[2:]
#
#     x /= 2
#     y /= 2
#
#     coordinate = np.array([x, y, z])
#
#     current_app.logger.debug(("before", coordinate))
#
#     if not cg.is_in_bounds(coordinate):
#         coordinate /= cg.segmentation_resolution
#
#         coordinate[0] *= 2
#         coordinate[1] *= 2
#
#     current_app.logger.debug(("after", coordinate))
#
#     atomic_id = cg.get_atomic_id_from_coord(coordinate[0],
#                                             coordinate[1],
#                                             coordinate[2],
#                                             parent_id=np.uint64(
#                                                 node_id),
#                                             remesh_preview=True)
#
#     if atomic_id is None:
#         raise cg_exceptions.BadRequest(
#             f"Could not determine supervoxel ID for coordinates {coordinate}.")
#
#     data_dict["id"].append(atomic_id)
#     data_dict["coord"].append(coordinate)
#
#     current_app.logger.debug(data_dict)
#     try:
#         new_roots = cg.shatter_nodes(user_id=user_id,
#                                      atomic_node_ids=data_dict['id'],
#                                      radius=radius)
#     except cg_exceptions.LockingError as e:
#         raise cg_exceptions.InternalServerError(
#             "Could not acquire root lock for shatter operation.")
#     except cg_exceptions.PreconditionError as e:
#         raise cg_exceptions.BadRequest(str(e))
#
#     if new_roots is None:
#         raise cg_exceptions.InternalServerError(
#             "Could not shatter selected region."
#         )
#
#     # Return binary
#     return app_utils.tobinary(new_roots)



### CHILDREN -------------------------------------------------------------------


@bp.route('/1.0/segment/<parent_id>/children', methods=['POST', 'GET'])
def handle_children_1(parent_id):
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
    return handle_children_main(table_id, parent_id)


@bp.route('/1.0/<table_id>/segment/<parent_id>/children',
          methods=['POST', 'GET'])
def handle_children_2(table_id, parent_id):
    return handle_children_main(table_id, parent_id)


def handle_children_main(table_id, parent_id):
    cg = app_utils.get_cg(table_id)

    parent_id = np.uint64(parent_id)
    layer = cg.get_chunk_layer(parent_id)

    if layer > 1:
        children = cg.get_children(parent_id)
    else:
        children = np.array([])

    # Return binary
    return app_utils.tobinary(children)


### LEAVES ---------------------------------------------------------------------


@bp.route('/1.0/segment/<root_id>/leaves', methods=['POST', 'GET'])
def handle_leaves_1(root_id):
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    return handle_leaves_main(table_id, root_id, bounding_box)


@bp.route('/1.0/<table_id>/segment/<root_id>/leaves', methods=['POST', 'GET'])
def handle_leaves_2(table_id, root_id):
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    return handle_leaves_main(table_id, root_id, bounding_box)


def handle_leaves_main(table_id, root_id, bounding_box):

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    atomic_ids = cg.get_subgraph_nodes(int(root_id),
                                       bounding_box=bounding_box,
                                       bb_is_coordinate=True)

    # Return binary
    return app_utils.tobinary(atomic_ids)


### LEAVES FROM LEAVES ---------------------------------------------------------

@bp.route('/1.0/segment/<atomic_id>/leaves_from_leave', methods=['POST', 'GET'])
def handle_leaves_from_leave_1(atomic_id):
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    return handle_leaves_from_leaves_main(table_id, atomic_id, bounding_box)


@bp.route('/1.0/<table_id>/segment/<atomic_id>/leaves_from_leave',
          methods=['POST', 'GET'])
def handle_leaves_from_leave_2(table_id, atomic_id):
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    return handle_leaves_from_leaves_main(table_id, atomic_id, bounding_box)


def handle_leaves_from_leaves_main(table_id, atomic_id, bounding_box):

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    root_id = cg.get_root(int(atomic_id))

    atomic_ids = cg.get_subgraph_nodes(root_id,
                                       bounding_box=bounding_box,
                                       bb_is_coordinate=True)
    # Return binary
    return app_utils.tobinary(np.concatenate([np.array([root_id]), atomic_ids]))


### SUBGRAPH -------------------------------------------------------------------


@bp.route('/1.0/segment/<root_id>/subgraph', methods=['POST', 'GET'])
def handle_subgraph_1(root_id):
    table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    return handle_subgraph_main(table_id, root_id, bounding_box)


@bp.route('/1.0/<table_id>/segment/<root_id>/subgraph', methods=['POST', 'GET'])
def handle_subgraph_2(table_id, root_id):
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    return handle_subgraph_main(table_id, root_id, bounding_box)


def handle_subgraph_main(table_id, root_id, bounding_box):

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    atomic_edges = cg.get_subgraph_edges(int(root_id),
                                         bounding_box=bounding_box,
                                         bb_is_coordinate=True)[0]
    # Return binary
    return app_utils.tobinary(atomic_edges)


### CHANGE LOG -----------------------------------------------------------------

@bp.route('/1.0/<table_id>/segment/<root_id>/change_log',
          methods=["POST", "GET"])
def change_log(table_id, root_id):
    try:
        time_stamp_past = float(request.args.get('timestamp', 0))
        time_stamp_past = datetime.fromtimestamp(time_stamp_past, UTC)
    except (TypeError, ValueError) as e:
        raise(cg_exceptions.BadRequest("Timestamp parameter is not a valid"
                                       " unix timestamp"))

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    change_log = cg.get_change_log(root_id=np.uint64(root_id),
                                   correct_for_wrong_coord_type=True,
                                   time_stamp_past=time_stamp_past)

    for k in change_log:
        data = change_log[k]
        if isinstance(data, np.ndarray):
            change_log[k] = app_utils.json_serialize_nd_array(data)

    return jsonify(change_log)

@bp.route('/1.0/<table_id>/segment/<root_id>/merge_log',
          methods=["POST", "GET"])
def merge_log(table_id, root_id):
    try:
        time_stamp_past = float(request.args.get('timestamp', 0))
        time_stamp_past = datetime.fromtimestamp(time_stamp_past, UTC)
    except (TypeError, ValueError) as e:
        raise(cg_exceptions.BadRequest("Timestamp parameter is not a valid"
                                       " unix timestamp"))

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    change_log = cg.get_change_log(root_id=np.uint64(root_id),
                                   correct_for_wrong_coord_type=True,
                                   time_stamp_past=time_stamp_past)

    for k in list(change_log.keys()):
        if not "merge" in k:
            del change_log[k]
            continue

        data = change_log[k]
        if isinstance(data, np.ndarray):
            change_log[k] = app_utils.json_serialize_nd_array(data)

    return jsonify(change_log)
