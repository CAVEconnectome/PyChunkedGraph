import collections
import json
import threading
import time
import traceback
import gzip
import os
from io import BytesIO as IO
from datetime import datetime
import requests
import networkx as nx
from scipy import spatial

import numpy as np
from pytz import UTC
import pandas as pd

from cloudvolume import compression

from middle_auth_client import get_usernames

from flask import current_app, g, jsonify, make_response, request
from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.backend import history as cg_history
from pychunkedgraph.backend.utils import column_keys
from pychunkedgraph.graph_analysis import analysis, contact_sites
from pychunkedgraph.backend.graphoperation import GraphEditOperation

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    NamedTuple,
)

__api_versions__ = [0, 1]
__segmentation_url_prefix__ = os.environ.get("SEGMENTATION_URL_PREFIX", "segmentation")


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
    current_app.user_id = None
    current_app.table_id = None
    current_app.request_type = None

    content_encoding = request.headers.get("Content-Encoding", "")

    if "gzip" in content_encoding.lower():
        request.data = compression.decompress(request.data, "gzip")


def after_request(response):
    dt = (time.time() - current_app.request_start_time) * 1000

    current_app.logger.debug("Response time: %.3fms" % dt)

    try:
        if current_app.user_id is None:
            user_id = ""
        else:
            user_id = current_app.user_id

        if current_app.table_id is not None:
            log_db = app_utils.get_log_db(current_app.table_id)
            log_db.add_success_log(
                user_id=user_id,
                user_ip="",
                request_time=current_app.request_start_date,
                response_time=dt,
                url=request.url,
                request_data=request.data,
                request_type=current_app.request_type,
            )
    except Exception as e:
        current_app.logger.debug(
            f"{current_app.user_id}: LogDB entry not" f" successful: {e}"
        )

    accept_encoding = request.headers.get("Accept-Encoding", "")

    if "gzip" not in accept_encoding.lower():
        return response

    response.direct_passthrough = False

    if (
        response.status_code < 200
        or response.status_code >= 300
        or "Content-Encoding" in response.headers
    ):
        return response

    response.data = compression.gzip_compress(response.data)

    response.headers["Content-Encoding"] = "gzip"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Content-Length"] = len(response.data)

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
    cg = app_utils.get_cg(table_id)

    dataset_info = cg.dataset_info
    app_info = {"app": {"supported_api_versions": list(__api_versions__)}}
    combined_info = {**dataset_info, **app_info}

    return jsonify(combined_info)


def handle_api_versions():
    return jsonify(__api_versions__)


### HELPERS -------------------------------------------------------------------
def handle_supervoxel_id_lookup(
    cg, coordinates: Sequence[Sequence[int]], node_ids: Sequence[np.uint64]
) -> Sequence[np.uint64]:
    """Helper to lookup supervoxel ids.

    This takes care of grouping coordinates."""

    def ccs(coordinates_nm_):
        graph = nx.Graph()

        dist_mat = spatial.distance.cdist(coordinates_nm_, coordinates_nm_)
        for edge in np.array(np.where(dist_mat < 1000)).T:
            graph.add_edge(*edge)

        ccs = [np.array(list(cc)) for cc in nx.connected_components(graph)]
        return ccs

    coordinates = np.array(coordinates, dtype=np.int)
    coordinates_nm = coordinates * cg.cv.resolution

    node_ids = np.array(node_ids, dtype=np.uint64)

    if len(coordinates.shape) != 2:
        raise cg_exceptions.BadRequest(
            f"Could not determine supervoxel ID for coordinates "
            f"{coordinates} - Validation stage."
        )

    atomic_ids = np.zeros(len(coordinates), dtype=np.uint64)
    for node_id in np.unique(node_ids):
        node_id_m = node_ids == node_id

        for cc in ccs(coordinates_nm[node_id_m]):
            m_ids = np.where(node_id_m)[0][cc]

            for max_dist_nm in [75, 150, 250, 500]:
                print(coordinates[m_ids], node_id)
                atomic_ids_sub = cg.get_atomic_ids_from_coords(
                    coordinates[m_ids], parent_id=node_id, max_dist_nm=max_dist_nm
                )
                if atomic_ids_sub is not None:
                    break
            if atomic_ids_sub is None:
                raise cg_exceptions.BadRequest(
                    f"Could not determine supervoxel ID for coordinates "
                    f"{coordinates} - Validation stage."
                )

            atomic_ids[m_ids] = atomic_ids_sub
    return atomic_ids


### GET ROOT -------------------------------------------------------------------


def handle_root(table_id, atomic_id):
    current_app.table_id = table_id

    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid unix timestamp"
            )
        )

    stop_layer = request.args.get("stop_layer", None)
    if stop_layer is not None:
        try:
            stop_layer = int(stop_layer)
        except (TypeError, ValueError):
            raise (cg_exceptions.BadRequest("stop_layer is not an integer"))

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    root_id = cg.get_root(
        np.uint64(atomic_id), stop_layer=stop_layer, time_stamp=timestamp
    )

    # Return root ID
    return root_id


### GET ROOTS -------------------------------------------------------------------


def handle_roots(table_id, is_binary=False):
    current_app.request_type = "roots"
    current_app.table_id = table_id

    if is_binary:
        node_ids = np.frombuffer(request.data, np.uint64)
    else:
        node_ids = np.array(json.loads(request.data)["node_ids"], dtype=np.uint64)
    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    stop_layer = request.args.get("stop_layer", None)
    if stop_layer is not None:
        stop_layer = int(stop_layer)
    assert_roots = bool(request.args.get("assert_roots", False))
    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    root_ids = cg.get_roots(
        node_ids, stop_layer=stop_layer, time_stamp=timestamp, assert_roots=assert_roots
    )

    return root_ids


### RANGE READ -------------------------------------------------------------------


def handle_l2_chunk_children(table_id, chunk_id, as_array):
    current_app.request_type = "l2_chunk_children"
    current_app.table_id = table_id

    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    chunk_layer = cg.get_chunk_layer(chunk_id)
    if chunk_layer != 2:
        raise (
            cg_exceptions.PreconditionError(
                f"This function only accepts level 2 chunks, the chunk requested is a level {chunk_layer} chunk"
            )
        )

    rr_chunk = cg.range_read_chunk(
        chunk_id=np.uint64(chunk_id),
        columns=column_keys.Hierarchy.Child,
        time_stamp=timestamp,
    )

    if as_array:
        l2_chunk_array = []

        for l2 in rr_chunk:
            svs = rr_chunk[l2][0].value
            for sv in svs:
                l2_chunk_array.extend([l2, sv])

        return np.array(l2_chunk_array)
    else:
        # store in dict of keys to arrays to remove reliance on bigtable
        l2_chunk_dict = {}
        for k in rr_chunk:
            l2_chunk_dict[k] = rr_chunk[k][0].value

        return l2_chunk_dict


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def trigger_remesh(table_id, new_lvl2_ids, is_priority=True):
    auth_header = {"Authorization": f"Bearer {current_app.config['AUTH_TOKEN']}"}
    resp = requests.post(
        f"{current_app.config['MESHING_ENDPOINT']}/api/v1/table/{table_id}/remeshing",
        data=json.dumps({"new_lvl2_ids": new_lvl2_ids}, cls=current_app.json_encoder),
        params={"priority": is_priority},
        headers=auth_header,
    )
    resp.raise_for_status()


### MERGE ----------------------------------------------------------------------


def handle_merge(table_id):
    current_app.table_id = table_id

    nodes = json.loads(request.data)
    is_priority = request.args.get("priority", True, type=str2bool)
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    current_app.logger.debug(nodes)
    assert len(nodes) == 2

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    node_ids = []
    coords = []
    for node in nodes:
        node_ids.append(node[0])
        coords.append(np.array(node[1:]) / cg.segmentation_resolution)

    atomic_edge = handle_supervoxel_id_lookup(cg, coords, node_ids)

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
        raise cg_exceptions.InternalServerError(
            "Could not merge selected " "supervoxel."
        )

    current_app.logger.debug(("lvl2_nodes:", ret.new_lvl2_ids))

    if len(ret.new_lvl2_ids) > 0:
        trigger_remesh(table_id, ret.new_lvl2_ids, is_priority=is_priority)

    return ret


### SPLIT ----------------------------------------------------------------------


def handle_split(table_id):
    current_app.table_id = table_id

    data = json.loads(request.data)
    is_priority = request.args.get("priority", True, type=str2bool)
    mincut = request.args.get("mincut", True, type=str2bool)
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    current_app.logger.debug(data)

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    node_idents = []
    coords = []
    node_ids = []

    for k in ["sources", "sinks"]:
        for node in data[k]:
            node_ids.append(node[0])
            coords.append(np.array(node[1:]) / cg.segmentation_resolution)
            node_idents.append(k)

    node_ids = np.array(node_ids, dtype=np.uint64)
    coords = np.array(coords)
    node_idents = np.array(node_idents)
    sv_ids = handle_supervoxel_id_lookup(cg, coords, node_ids)

    current_app.logger.debug(
        {"node_id": node_ids, "sv_id": sv_ids, "node_ident": node_idents}
    )

    try:
        ret = cg.remove_edges(
            user_id=user_id,
            source_ids=sv_ids[node_idents == "sources"],
            sink_ids=sv_ids[node_idents == "sinks"],
            source_coords=coords[node_idents == "sources"],
            sink_coords=coords[node_idents == "sinks"],
            mincut=mincut,
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
        trigger_remesh(table_id, ret.new_lvl2_ids, is_priority=is_priority)

    return ret


### UNDO ----------------------------------------------------------------------


def handle_undo(table_id):
    if table_id in ["fly_v26", "fly_v31"]:
        raise cg_exceptions.InternalServerError(
            "Undo not supported for this chunkedgraph table."
        )

    current_app.table_id = table_id

    data = json.loads(request.data)
    is_priority = request.args.get("priority", True, type=str2bool)
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

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
        trigger_remesh(table_id, ret.new_lvl2_ids, is_priority=is_priority)

    return ret


### REDO ----------------------------------------------------------------------


def handle_redo(table_id):
    if table_id in ["fly_v26", "fly_v31"]:
        raise cg_exceptions.InternalServerError(
            "Redo not supported for this chunkedgraph table."
        )

    current_app.table_id = table_id

    data = json.loads(request.data)
    is_priority = request.args.get("priority", True, type=str2bool)
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

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
        trigger_remesh(table_id, ret.new_lvl2_ids, is_priority=is_priority)

    return ret


### ROLLBACK USER --------------------------------------------------------------


def handle_rollback(table_id):
    if table_id in ["fly_v26", "fly_v31"]:
        raise cg_exceptions.InternalServerError(
            "Rollback not supported for this chunkedgraph table."
        )

    current_app.table_id = table_id

    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id
    target_user_id = request.args["user_id"]

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    user_operations = all_user_operations(table_id)
    operation_ids = user_operations["operation_id"]
    timestamps = user_operations["timestamp"]
    operations = list(zip(operation_ids, timestamps))
    operations.sort(key=lambda op: op[1], reverse=True)

    for operation in operations:
        operation_id = operation[0]
        try:
            ret = cg.undo_operation(user_id=target_user_id, operation_id=operation_id)
        except cg_exceptions.LockingError as e:
            raise cg_exceptions.InternalServerError(
                "Could not acquire root lock for undo operation."
            )
        except (cg_exceptions.PreconditionError, cg_exceptions.PostconditionError) as e:
            raise cg_exceptions.BadRequest(str(e))

        if ret.new_lvl2_ids.size > 0:
            trigger_remesh(table_id, ret.new_lvl2_ids, is_priority=False)

    return user_operations


### USER OPERATIONS -------------------------------------------------------------


def all_user_operations(table_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id
    target_user_id = request.args["user_id"]

    try:
        start_time = float(request.args.get("start_time", 0))
        start_time = datetime.fromtimestamp(start_time, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "start_time parameter is not a valid unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg_instance = app_utils.get_cg(table_id)

    log_rows = cg_instance.read_log_rows(start_time=start_time)

    valid_entry_ids = []
    timestamp_list = []

    entry_ids = np.sort(list(log_rows.keys()))
    for entry_id in entry_ids:
        entry = log_rows[entry_id]
        user_id = entry[column_keys.OperationLogs.UserID]

        if user_id == target_user_id:
            valid_entry_ids.append(entry_id)
            timestamp = entry["timestamp"]
            timestamp_list.append(timestamp)

    return {"operation_id": valid_entry_ids, "timestamp": timestamp_list}


### CHILDREN -------------------------------------------------------------------


def handle_children(table_id, parent_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

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
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id
    stop_layer = int(request.args.get("stop_layer", 1))
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    if stop_layer > 1:
        subgraph = cg.get_subgraph_nodes(
            int(root_id),
            bounding_box=bounding_box,
            bb_is_coordinate=True,
            return_layers=[stop_layer],
        )
        if isinstance(subgraph, np.ndarray):
            return subgraph
        else:
            empty_1d = np.empty(0, dtype=np.uint64)
            result = [empty_1d]
            for node_subgraph in subgraph.values():
                for children_at_layer in node_subgraph.values():
                    result.append(children_at_layer)
            return np.concatenate(result)
    else:
        atomic_ids = cg.get_subgraph_nodes(
            int(root_id), bounding_box=bounding_box, bb_is_coordinate=True
        )

        return atomic_ids


### LEAVES OF MANY ROOTS ---------------------------------------------------------------------


def handle_leaves_many(table_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    stop_layer = int(request.args.get("stop_layer", 1))
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array(
            [b.split("-") for b in bounds.split("_")], dtype=np.int
        ).T
    else:
        bounding_box = None

    node_ids = np.array(json.loads(request.data)["node_ids"], dtype=np.uint64)

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    node_to_leaves_mapping = cg.get_subgraph_nodes(
        node_ids,
        bounding_box=bounding_box,
        bb_is_coordinate=True,
        return_layers=[stop_layer],
        serializable=True,
    )

    return node_to_leaves_mapping


### LEAVES FROM LEAVES ---------------------------------------------------------


def handle_leaves_from_leave(table_id, atomic_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

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
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

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
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    try:
        time_stamp_past = float(request.args.get("timestamp", 0))
        time_stamp_past = datetime.fromtimestamp(time_stamp_past, UTC)
    except (TypeError, ValueError):
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


def tabular_change_log_recent(table_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    try:
        start_time = float(request.args.get("start_time", 0))
        start_time = datetime.fromtimestamp(start_time, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "start_time parameter is not a valid unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg_instance = app_utils.get_cg(table_id)

    log_rows = cg_instance.read_log_rows(start_time=start_time)

    timestamp_list = []
    user_list = []

    entry_ids = np.sort(list(log_rows.keys()))
    for entry_id in entry_ids:
        entry = log_rows[entry_id]

        timestamp = entry["timestamp"]
        timestamp_list.append(timestamp)

        user_id = entry[column_keys.OperationLogs.UserID]
        user_list.append(user_id)

    return pd.DataFrame.from_dict(
        {"operation_id": entry_ids, "timestamp": timestamp_list, "user_id": user_list}
    )


def tabular_change_log(table_id, root_id, get_root_ids, filtered):
    if get_root_ids:
        current_app.request_type = "tabular_changelog_wo_ids"
    else:
        current_app.request_type = "tabular_changelog"

    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    segment_history = cg_history.SegmentHistory(cg, int(root_id))

    tab = segment_history.get_tabular_changelog(
        with_ids=get_root_ids, filtered=filtered
    )

    try:
        tab["user_name"] = get_usernames(
            np.array(tab["user_id"], dtype=np.int).squeeze(),
            current_app.config["AUTH_TOKEN"],
        )
    except:
        current_app.logger.error(f"Could not retrieve user names for {root_id}")

    return tab


def merge_log(table_id, root_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    try:
        time_stamp_past = float(request.args.get("timestamp", 0))
        time_stamp_past = datetime.fromtimestamp(time_stamp_past, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    segment_history = cg_history.SegmentHistory(cg, int(root_id))
    return segment_history.merge_log(correct_for_wrong_coord_type=True)


def handle_lineage_graph(table_id, root_id=None):
    from networkx import node_link_data

    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    def _parse_timestamp(arg_name, default_timestamp=0):
        """Convert seconds since epoch to UTC datetime."""
        try:
            return float(request.args.get(arg_name, default_timestamp))
        except (TypeError, ValueError):
            raise (
                cg_exceptions.BadRequest(
                    "Timestamp parameter is not a valid unix timestamp"
                )
            )

    timestamp_past = _parse_timestamp("timestamp_past")
    timestamp_future = _parse_timestamp("timestamp_future", time.time())

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    if root_id is None:
        from ...backend.lineage import lineage_graph

        root_ids = np.array(json.loads(request.data)["root_ids"], dtype=np.uint64)
        graph = lineage_graph(cg, root_ids, timestamp_past, timestamp_future)
        return node_link_data(graph)
    graph = cg_history.SegmentHistory(cg, int(root_id)).get_change_log_graph(
        timestamp_past, timestamp_future
    )
    return node_link_data(graph)


def handle_past_id_mapping(table_id):
    root_ids = np.array(json.loads(request.data)["root_ids"], dtype=np.uint64)
    try:
        timestamp_past = float(request.args.get("timestamp_past", 0))
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid unix timestamp"
            )
        )

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    past_id_mapping = {}
    future_id_mapping = {}
    for root_id in root_ids:
        hist = cg_history.SegmentHistory(cg, int(root_id))
        graph = hist.get_change_log_graph(timestamp_past, None)

        in_degree_dict = dict(graph.in_degree)
        nodes = np.array(list(in_degree_dict.keys()))
        in_degrees = np.array(list(in_degree_dict.values()))

        past_id_mapping[int(root_id)] = nodes[in_degrees == 0]

    return {"past_id_map": past_id_mapping, "future_id_map": future_id_mapping}


def last_edit(table_id, root_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    cg = app_utils.get_cg(table_id)

    segment_history = cg_history.SegmentHistory(cg, int(root_id))

    return segment_history.last_edit.timestamp


def oldest_timestamp(table_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

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

    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError):
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

    cs_list, cs_metadata = contact_sites.get_contact_sites(
        cg,
        np.uint64(root_id),
        bounding_box=bounding_box,
        compute_partner=partners,
        end_time=timestamp,
        as_list=as_list,
        areas_only=areas_only,
    )

    return cs_list, cs_metadata


def handle_pairwise_contact_sites(table_id, first_node_id, second_node_id):
    current_app.request_type = "pairwise_contact_sites"
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )
    exact_location = request.args.get("exact_location", True, type=app_utils.toboolean)
    cg = app_utils.get_cg(table_id)
    contact_sites_list, cs_metadata = contact_sites.get_contact_sites_pairwise(
        cg,
        np.uint64(first_node_id),
        np.uint64(second_node_id),
        end_time=timestamp,
        exact_location=exact_location,
    )
    return contact_sites_list, cs_metadata


### SPLIT PREVIEW --------------------------------------------------------------


def handle_split_preview(table_id):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    data = json.loads(request.data)
    current_app.logger.debug(data)

    cg = app_utils.get_cg(table_id)

    node_idents = []
    coords = []
    node_ids = []

    for k in ["sources", "sinks"]:
        for node in data[k]:
            node_ids.append(node[0])
            coords.append(np.array(node[1:]) / cg.segmentation_resolution)
            node_idents.append(k)

    node_ids = np.array(node_ids, dtype=np.uint64)
    coords = np.array(coords, dtype=np.int)
    node_idents = np.array(node_idents)
    sv_ids = handle_supervoxel_id_lookup(cg, coords, node_ids)

    current_app.logger.debug(
        {"node_id": node_ids, "sv_id": sv_ids, "node_ident": node_idents}
    )

    try:
        supervoxel_ccs, illegal_split = cg._run_multicut(
            source_ids=sv_ids[node_idents == "sources"],
            sink_ids=sv_ids[node_idents == "sinks"],
            source_coords=coords[node_idents == "sources"],
            sink_coords=coords[node_idents == "sinks"],
            bb_offset=(240, 240, 24),
            split_preview=True,
        )

    except cg_exceptions.PreconditionError as e:
        raise cg_exceptions.BadRequest(str(e))

    resp = {
        "supervoxel_connected_components": supervoxel_ccs,
        "illegal_split": illegal_split,
    }
    return resp


### FIND PATH --------------------------------------------------------------


def handle_find_path(table_id, precision_mode):
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    nodes = json.loads(request.data)

    current_app.logger.debug(nodes)
    assert len(nodes) == 2

    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)
    node_ids = []
    coords = []
    for node in nodes:
        node_ids.append(node[0])
        coords.append(np.array(node[1:]) / cg.segmentation_resolution)

    source_supervoxel_id, target_supervoxel_id = handle_supervoxel_id_lookup(
        cg, coords, node_ids
    )

    source_l2_id = cg.get_parent(source_supervoxel_id)
    target_l2_id = cg.get_parent(target_supervoxel_id)

    l2_path = analysis.find_l2_shortest_path(cg, source_l2_id, target_l2_id)
    if precision_mode:
        centroids, failed_l2_ids = analysis.compute_mesh_centroids_of_l2_ids(
            cg, l2_path, flatten=True
        )
        return {
            "centroids_list": centroids,
            "failed_l2_ids": failed_l2_ids,
            "l2_path": l2_path,
        }
    else:
        centroids = analysis.compute_rough_coordinate_path(cg, l2_path)
        return {"centroids_list": centroids, "failed_l2_ids": [], "l2_path": l2_path}


### GET_LAYER2_SUBGRAPH
def handle_get_layer2_graph(table_id, node_id):
    current_app.request_type = "get_lvl2_graph"
    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id

    cg = app_utils.get_cg(table_id)
    edge_graph = analysis.get_lvl2_edge_list(cg, int(node_id))
    return {"edge_graph": edge_graph}


### IS LATEST ROOTS --------------------------------------------------------------


def handle_is_latest_roots(table_id, is_binary):
    current_app.request_type = "is_latest_roots"
    current_app.table_id = table_id

    if is_binary:
        node_ids = np.frombuffer(request.data, np.uint64)
    else:
        node_ids = np.array(json.loads(request.data)["node_ids"], dtype=np.uint64)
    # Convert seconds since epoch to UTC datetime
    try:
        timestamp = float(request.args.get("timestamp", time.time()))
        timestamp = datetime.fromtimestamp(timestamp, UTC)
    except (TypeError, ValueError):
        raise (
            cg_exceptions.BadRequest(
                "Timestamp parameter is not a valid" " unix timestamp"
            )
        )
    # Call ChunkedGraph
    cg = app_utils.get_cg(table_id)

    row_dict = cg.read_node_id_rows(
        node_ids=node_ids, columns=column_keys.Hierarchy.NewParent, end_time=timestamp
    )

    if not np.all(cg.get_chunk_layers(node_ids) == cg.n_layers):
        raise cg_exceptions.BadRequest("Some ids are not root ids.")

    is_latest = ~np.isin(node_ids, list(row_dict.keys()))

    return is_latest


### OPERATION DETAILS ------------------------------------------------------------


def operation_details(table_id):
    def parse_attr(attr, val) -> str:
        from numpy import ndarray

        try:
            if isinstance(val, ndarray):
                return (attr.key, val.tolist())
            return (attr.key, val)
        except AttributeError:
            return (attr, val)

    current_app.table_id = table_id
    user_id = str(g.auth_user["id"])
    current_app.user_id = user_id
    operation_ids = json.loads(request.args["operation_ids"])

    cg = app_utils.get_cg(table_id)

    log_rows = cg.read_log_rows(operation_ids)

    result = {}
    for k, v in log_rows.items():
        details = {}
        for _k, _v in v.items():
            _k, _v = parse_attr(_k, _v)
            try:
                details[_k.decode("utf-8")] = _v
            except AttributeError:
                details[_k] = _v
        result[int(k)] = details
    return result
