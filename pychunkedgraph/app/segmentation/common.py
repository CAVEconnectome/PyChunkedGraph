import collections
import json
import threading
import time
import traceback
from datetime import datetime

import numpy as np
from pytz import UTC

from flask import current_app, g, jsonify, make_response, request
from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.app.meshing.common import _remeshing
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.backend import history as cg_history
from pychunkedgraph.graph_analysis import analysis, contact_sites

__api_versions__ = [0, 1]


def index():
    return f"PyChunkedGraph Segmentation v{__version__}"


def home():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


def before_request():
    current_app.request_start_time = time.time()
    current_app.request_start_date = datetime.utcnow()


def after_request(response):
    dt = (time.time() - current_app.request_start_time) * 1000

    current_app.logger.debug("Response time: %.3fms" % dt)

    try:
        log_db = app_utils.get_log_db(current_app.table_id)
        log_db.add_success_log(
            user_id="",
            user_ip="",
            request_time=current_app.request_start_date,
            response_time=dt,
            url=request.url,
            request_data=request.data,
            request_type=current_app.request_type,
        )
    except:
        current_app.logger.debug("LogDB entry not successful")

    return response


def unhandled_exception(e):
    status_code = 500
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": status_code,
            "traceback": tb,
        }
    )

    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": status_code,
        "message": str(e),
        "traceback": tb,
    }

    return jsonify(resp), status_code


def api_exception(e):
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": e.status_code.value,
            "traceback": tb,
        }
    )

    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": e.status_code.value,
        "message": str(e),
    }

    return jsonify(resp), e.status_code.value


# -------------------
# ------ Applications
# -------------------


def sleep_me(sleep):
    current_app.request_type = "sleep"

    time.sleep(sleep)
    return "zzz... {} ... awake".format(sleep)


def handle_info(table_id):
    current_app.request_type = "info"

    cg = app_utils.get_cg(table_id)

    dataset_info = cg.dataset_info
    app_info = {"app": {"supported_api_versions": list(__api_versions__)}}
    combined_info = {**dataset_info, **app_info}

    return jsonify(combined_info)


def handle_api_versions():
    return jsonify(__api_versions__)


### GET ROOT -------------------------------------------------------------------


def handle_root(table_id, atomic_id):
    current_app.request_type = "root"

    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError) as e:
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    root_id = cg.get_root(np.uint64(atomic_id), time_stamp=timestamp)

    # Return root ID
    return root_id


### MERGE ----------------------------------------------------------------------


def handle_merge(table_id):
    current_app.request_type = "merge"

    nodes = json.loads(request.data)
    user_id = str(g.auth_user["id"])

    current_app.logger.debug(nodes)
    assert len(nodes) == 2

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    atomic_edge = []
    coords = []
    for node in nodes:
        node_id = node[0]
        x, y, z = node[1:]
        coordinate = np.array([x, y, z]) / cg.segmentation_resolution

        atomic_id = cg.get_atomic_id_from_coord(
            coordinate[0], coordinate[1], coordinate[2], parent_id=np.uint64(node_id)
        )

        if atomic_id is None:
            raise cg_exceptions.BadRequest(
                f"Could not determine supervoxel ID for coordinates " f"{coordinate}."
            )

        coords.append(coordinate)
        atomic_edge.append(atomic_id)

    # Protection from long range mergers
    chunk_coord_delta = cg.get_chunk_coordinates(
        atomic_edge[0]
    ) - cg.get_chunk_coordinates(atomic_edge[1])

    if np.any(np.abs(chunk_coord_delta) > 3):
        raise cg_exceptions.BadRequest(
            "Chebyshev distance between merge points exceeded allowed maximum "
            "(3 chunks)."
        )

    try:
        ret = cg.add_edges(
            user_id=user_id,
            atomic_edges=np.array(atomic_edge, dtype=np.uint64),
            source_coord=coords[:1],
            sink_coord=coords[1:],
        )

    except cg_exceptions.LockingError as e:
        raise cg_exceptions.InternalServerError(
            "Could not acquire root lock for merge operation."
        )
    except cg_exceptions.PreconditionError as e:
        raise cg_exceptions.BadRequest(str(e))

    if ret.new_root_ids is None:
        raise cg_exceptions.InternalServerError("Could not merge selected supervoxel.")

    current_app.logger.debug(("lvl2_nodes:", ret.new_lvl2_ids))

    if len(ret.new_lvl2_ids) > 0:
        t = threading.Thread(
            target=_remeshing, args=(cg.get_serialized_info(), ret.new_lvl2_ids)
        )
        t.start()

    return ret


### SPLIT ----------------------------------------------------------------------


def handle_split(table_id):
    current_app.request_type = "split"

    data = json.loads(request.data)
    user_id = str(g.auth_user["id"])

    current_app.logger.debug(data)

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    data_dict = {}
    for k in ["sources", "sinks"]:
        data_dict[k] = collections.defaultdict(list)

        for node in data[k]:
            node_id = node[0]
            x, y, z = node[1:]
            coordinate = np.array([x, y, z]) / cg.segmentation_resolution

            atomic_id = cg.get_atomic_id_from_coord(
                coordinate[0],
                coordinate[1],
                coordinate[2],
                parent_id=np.uint64(node_id),
            )

            if atomic_id is None:
                raise cg_exceptions.BadRequest(
                    f"Could not determine supervoxel ID for coordinates "
                    f"{coordinate}."
                )

            data_dict[k]["id"].append(atomic_id)
            data_dict[k]["coord"].append(coordinate)

    current_app.logger.debug(data_dict)

    try:
        ret = cg.remove_edges(
            user_id=user_id,
            source_ids=data_dict["sources"]["id"],
            sink_ids=data_dict["sinks"]["id"],
            source_coords=data_dict["sources"]["coord"],
            sink_coords=data_dict["sinks"]["coord"],
            mincut=True,
        )

    except cg_exceptions.LockingError as e:
        raise cg_exceptions.InternalServerError(
            "Could not acquire root lock for split operation."
        )
    except cg_exceptions.PreconditionError as e:
        raise cg_exceptions.BadRequest(str(e))

    if ret.new_root_ids is None:
        raise cg_exceptions.InternalServerError(
            "Could not split selected segment groups."
        )

    current_app.logger.debug(("after split:", ret.new_root_ids))
    current_app.logger.debug(("lvl2_nodes:", ret.new_lvl2_ids))

    if len(ret.new_lvl2_ids) > 0:
        t = threading.Thread(
            target=_remeshing, args=(cg.get_serialized_info(), ret.new_lvl2_ids)
        )
        t.start()

    return ret


### UNDO ----------------------------------------------------------------------


def handle_undo(table_id):
    current_app.request_type = "undo"

    data = json.loads(request.data)
    user_id = str(g.auth_user["id"])

    current_app.logger.debug(data)

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    operation_id = np.uint64(data["operation_id"])

    try:
        ret = cg.undo(user_id=user_id, operation_id=operation_id)
    except cg_exceptions.LockingError as e:
        raise cg_exceptions.InternalServerError(
            "Could not acquire root lock for undo operation."
        )
    except (cg_exceptions.PreconditionError, cg_exceptions.PostconditionError) as e:
        raise cg_exceptions.BadRequest(str(e))

    current_app.logger.debug(("after undo:", ret.new_root_ids))
    current_app.logger.debug(("lvl2_nodes:", ret.new_lvl2_ids))

    if ret.new_lvl2_ids.size > 0:
        t = threading.Thread(
            target=_remeshing, args=(cg.get_serialized_info(), ret.new_lvl2_ids)
        )
        t.start()

    return ret


### REDO ----------------------------------------------------------------------


def handle_redo(table_id):
    current_app.request_type = "redo"

    data = json.loads(request.data)
    user_id = str(g.auth_user["id"])

    current_app.logger.debug(data)

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    operation_id = np.uint64(data["operation_id"])

    try:
        ret = cg.redo(user_id=user_id, operation_id=operation_id)
    except cg_exceptions.LockingError as e:
        raise cg_exceptions.InternalServerError(
            "Could not acquire root lock for redo operation."
        )
    except (cg_exceptions.PreconditionError, cg_exceptions.PostconditionError) as e:
        raise cg_exceptions.BadRequest(str(e))

    current_app.logger.debug(("after redo:", ret.new_root_ids))
    current_app.logger.debug(("lvl2_nodes:", ret.new_lvl2_ids))

    if ret.new_lvl2_ids.size > 0:
        t = threading.Thread(
            target=_remeshing, args=(cg.get_serialized_info(), ret.new_lvl2_ids)
        )
        t.start()

    return ret


### CHILDREN -------------------------------------------------------------------


def handle_children(table_id, parent_id):
    current_app.request_type = "children"

    cg = app_utils.get_cg(table_id)

    parent_id = np.uint64(parent_id)
    layer = cg.get_chunk_layer(parent_id)

    if layer > 1:
        children = cg.get_children(parent_id)
    else:
        children = np.array([])

    return children


### LEAVES ---------------------------------------------------------------------


def handle_leaves(table_id, root_id):
    current_app.request_type = "leaves"

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    atomic_ids = cg.get_subgraph_nodes(
        int(root_id), bounding_box=bounding_box, bb_is_coordinate=True
    )

    return atomic_ids


### LEAVES FROM LEAVES ---------------------------------------------------------


def handle_leaves_from_leave(table_id, atomic_id):
    current_app.request_type = "leaves_from_leave"

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    root_id = cg.get_root(int(atomic_id))

    atomic_ids = cg.get_subgraph_nodes(
        root_id, bounding_box=bounding_box, bb_is_coordinate=True
    )

    return np.concatenate([np.array([root_id]), atomic_ids])


### SUBGRAPH -------------------------------------------------------------------


def handle_subgraph(table_id, root_id):
    current_app.request_type = "subgraph"

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    atomic_edges = cg.get_subgraph_edges(
        int(root_id), bounding_box=bounding_box, bb_is_coordinate=True
    )[0]

    return atomic_edges


### CHANGE LOG -----------------------------------------------------------------


def change_log(table_id, root_id=None):
    current_app.request_type = "change_log"

    try:
        time_stamp_past = float(request.args.get("timestamp", 0))
        time_stamp_past = datetime.fromtimestamp(time_stamp_past, UTC)
    except (TypeError, ValueError) as e:
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    if not root_id:
        return cg_history.get_all_log_entries(cg)

    segment_history = cg_history.SegmentHistory(cg, int(root_id))

    return segment_history.change_log()


def merge_log(table_id, root_id):
    current_app.request_type = "merge_log"

    try:
        time_stamp_past = float(request.args.get("timestamp", 0))
        time_stamp_past = datetime.fromtimestamp(time_stamp_past, UTC)
    except (TypeError, ValueError) as e:
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    segment_history = cg_history.SegmentHistory(cg, int(root_id))
    return segment_history.merge_log(correct_for_wrong_coord_type=True)


def last_edit(table_id, root_id):
    current_app.request_type = "last_edit"

    cg = app_utils.get_cg(table_id)

    segment_history = cg_history.SegmentHistory(cg, int(root_id))

    return segment_history.last_edit.timestamp


def oldest_timestamp(table_id):
    current_app.request_type = "timestamp"

    cg = app_utils.get_cg(table_id)

    try:
        earliest_timestamp = cg.get_earliest_timestamp()
    except cg_exceptions.PreconditionError:
        raise cg_exceptions.InternalServerError("No timestamp available")

    return earliest_timestamp


### CONTACT SITES --------------------------------------------------------------


def handle_contact_sites(table_id, root_id):
    partners = request.args.get("partners", True, type=app_utils.toboolean)
    as_list = request.args.get("as_list", True, type=app_utils.toboolean)
    areas_only = request.args.get("areas_only", True, type=app_utils.toboolean)

    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError) as e:
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    cs_list = contact_sites.get_contact_sites(
        cg,
        np.uint64(root_id),
        bounding_box=bounding_box,
        compute_partner=partners,
        end_time=timestamp,
        as_list=as_list,
        areas_only=areas_only
    )

    return cs_list

def handle_pairwise_contact_sites(table_id, first_node_id, second_node_id):
    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError) as e:
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )
    exact_location = request.args.get("exact_location", True, type=app_utils.toboolean)
    cg = app_utils.get_cg(table_id)
    contact_sites_list = contact_sites.get_contact_sites_pairwise(
        cg,
        np.uint64(first_node_id),
        np.uint64(second_node_id),
        end_time=timestamp,
        exact_location=exact_location,
    )
    return contact_sites_list


### SPLIT PREVIEW --------------------------------------------------------------


def handle_split_preview(table_id):
    current_app.request_type = "split_preview"

    data = json.loads(request.data)
    current_app.logger.debug(data)

    cg = app_utils.get_cg(table_id)

    data_dict = {}
    for k in ["sources", "sinks"]:
        data_dict[k] = collections.defaultdict(list)

        for node in data[k]:
            node_id = node[0]
            x, y, z = node[1:]
            coordinate = np.array([x, y, z]) / cg.segmentation_resolution

            atomic_id = cg.get_atomic_id_from_coord(coordinate[0],
                                                    coordinate[1],
                                                    coordinate[2],
                                                    parent_id=np.uint64(
                                                        node_id))

            if atomic_id is None:
                raise cg_exceptions.BadRequest(
                    f"Could not determine supervoxel ID for coordinates "
                    f"{coordinate}.")

            data_dict[k]["id"].append(atomic_id)
            data_dict[k]["coord"].append(coordinate)

    current_app.logger.debug(data_dict)

    try:
        supervoxel_ccs, illegal_split = cg._run_multicut(
            source_ids=data_dict["sources"]["id"],
            sink_ids=data_dict["sinks"]["id"],
            source_coords=data_dict["sources"]["coord"],
            sink_coords=data_dict["sinks"]["coord"],
            bb_offset=(240,240,24),
            split_preview=True
        )

    except cg_exceptions.PreconditionError as e:
        raise cg_exceptions.BadRequest(str(e))

    resp = {
        "supervoxel_connected_components": supervoxel_ccs,
        "illegal_split": illegal_split
        }
    return resp


### FIND PATH --------------------------------------------------------------


def handle_find_path(table_id):
    current_app.request_type = "find_path"

    nodes = json.loads(request.data)

    current_app.logger.debug(nodes)
    assert len(nodes) == 2

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    def _get_supervoxel_id_from_node(node):
        node_id = node[0]
        x, y, z = node[1:]
        coordinate = np.array([x, y, z]) / cg.segmentation_resolution

        supervoxel_id = cg.get_atomic_id_from_coord(coordinate[0],
                                                coordinate[1],
                                                coordinate[2],
                                                parent_id=np.uint64(node_id))
        if supervoxel_id is None:
            raise cg_exceptions.BadRequest(
                f"Could not determine supervoxel ID for coordinates "
                f"{coordinate}."
            )

        return supervoxel_id

    source_supervoxel_id = _get_supervoxel_id_from_node(nodes[0])
    target_supervoxel_id = _get_supervoxel_id_from_node(nodes[1])
    source_l2_id = cg.get_parent(source_supervoxel_id)
    target_l2_id = cg.get_parent(target_supervoxel_id)

    l2_path = analysis.find_l2_shortest_path(cg, source_l2_id, target_l2_id)
    centroids, failed_l2_ids = analysis.compute_mesh_centroids_of_l2_ids(cg, l2_path, flatten=True)

    return {
        "centroids_list": centroids,
        "failed_l2_ids": failed_l2_ids,
        "l2_path": l2_path
    }
