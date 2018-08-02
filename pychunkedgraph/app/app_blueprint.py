from flask import Blueprint, request, make_response, g
from flask import current_app
# from google.cloud import pubsub_v1
import json
import numpy as np
import time
import datetime
# import pymongo

from pychunkedgraph.app import app_utils


bp = Blueprint('pychunkedgraph', __name__, url_prefix="/segmentation/")

# -------------------------------
# ------ Access control and index
# -------------------------------

@bp.route('/')
@bp.route("/index")
def index():
    return "PyChunkedGraph Server -- 0.2"


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
    print("NEW REQUEST:", datetime.datetime.now(), request.url)
    g.request_start_time = time.time()


@bp.after_request
def after_request(response):
    dt = (time.time() - g.request_start_time) * 1000

    url_split = request.url.split("/")
    current_app.logger.info("%s - %s - %s - %s - %f.3" %
                            (request.path.split("/")[-1], "1",
                             "".join([url_split[-2], "/", url_split[-1]]),
                             str(request.data), dt))

    print("Response time: %.3fms" % (dt))
    return response


@bp.errorhandler(500)
def internal_server_error(error):
    dt = (time.time() - g.request_start_time) * 1000

    url_split = request.url.split("/")
    current_app.logger.error("%s - %s - %s - %s - %f.3" %
                             (request.path.split("/")[-1],
                              "Server Error: " + error,
                              "".join([url_split[-2], "/", url_split[-1]]),
                              str(request.data), dt))
    return 500


@bp.errorhandler(Exception)
def unhandled_exception(e):
    dt = (time.time() - g.request_start_time) * 1000

    url_split = request.url.split("/")
    current_app.logger.error("%s - %s - %s - %s - %f.3" %
                             (request.path.split("/")[-1],
                              "Exception: " + str(e),
                              "".join([url_split[-2], "/", url_split[-1]]),
                              str(request.data), dt))
    return 500

# -------------------
# ------ Applications
# -------------------


@bp.route('/1.0/graph/root', methods=['POST', 'GET'])
def handle_root():
    atomic_id = int(json.loads(request.data)[0])

    # Call ChunkedGraph
    cg = app_utils.get_cg()
    root_id = cg.get_root(atomic_id)

    # Return binary
    return app_utils.tobinary(root_id)


@bp.route('/1.0/graph/merge', methods=['POST', 'GET'])
def handle_merge():
    nodes = json.loads(request.data)

    assert len(nodes) == 2

    user_id = str(request.remote_addr)

    # Call ChunkedGraph
    cg = app_utils.get_cg()

    atomic_edge = []
    for node in nodes:
        node_id = node[0]
        x, y, z = node[1:]

        atomic_id = cg.get_atomic_id_from_coord(x, y, z,
                                                parent_id=np.uint64(node_id))
        if atomic_id is None:
            return None

        atomic_edge.append(atomic_id)

    # Protection from long range mergers
    chunk_coord_delta = cg.get_chunk_coordinates(atomic_edge[0]) - \
                        cg.get_chunk_coordinates(atomic_edge[1])

    if np.any(np.abs(chunk_coord_delta) > 1):
        return None

    new_root = cg.add_edge(user_id=user_id,
                           atomic_edge=np.array(atomic_edge, dtype=np.uint64))

    if new_root is None:
        return None

    # Return binary
    return app_utils.tobinary(new_root)


@bp.route('/1.0/graph/split', methods=['POST', 'GET'])
def handle_split():
    data = json.loads(request.data)

    user_id = str(request.remote_addr)

    # Call ChunkedGraph
    cg = app_utils.get_cg()

    data_dict = {}
    for k in ["sources", "sinks"]:
        data_dict[k] = []

        for node in data[k]:
            node_id = node[0]
            x, y, z = node[1:]

            atomic_id = cg.get_atomic_id_from_coord(x, y, z,
                                                    parent_id=np.uint64(node_id))
            if atomic_id is None:
                return None

            data_dict[k].append({"id": atomic_id,
                                 "coord": np.array([x, y, z])})

    print(data_dict)
    new_roots = cg.remove_edges(user_id=user_id,
                                source_id=data_dict["sources"][0]["id"],
                                sink_id=data_dict["sinks"][0]["id"],
                                source_coord=data_dict["sources"][0]["coord"],
                                sink_coord=data_dict["sinks"][0]["coord"],
                                mincut=True)

    if new_roots is None:
        return None

    # Return binary
    return app_utils.tobinary(new_roots)


@bp.route('/1.0/segment/<parent_id>/children', methods=['POST', 'GET'])
def handle_children(parent_id):
    # Call ChunkedGraph
    cg = app_utils.get_cg()

    parent_id = np.uint64(parent_id)
    layer = cg.get_chunk_layer(parent_id)

    if layer > 4:
        stop_lvl = 4
    elif layer > 3:
        stop_lvl = 3
    elif layer == 3:
        stop_lvl = 2
    else:
        stop_lvl = 1

    try:
        children = cg.get_subgraph(parent_id, stop_lvl=stop_lvl)
    except:
        children = np.array([])

    # Return binary
    return app_utils.tobinary(children)


@bp.route('/1.0/segment/<root_id>/leaves', methods=['POST', 'GET'])
def handle_leaves(root_id):
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg()
    atomic_ids = cg.get_subgraph(int(root_id),
                                 bounding_box=bounding_box,
                                 bb_is_coordinate=True)

    # Return binary
    return app_utils.tobinary(atomic_ids)


@bp.route('/1.0/segment/<atomic_id>/leaves_from_leave',
          methods=['POST', 'GET'])
def handle_leaves_from_leave(atomic_id):
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg()
    root_id = cg.get_root(int(atomic_id))

    atomic_ids = cg.get_subgraph(root_id,
                                 bounding_box=bounding_box,
                                 bb_is_coordinate=True)
    # Return binary
    return app_utils.tobinary(np.concatenate([np.array([root_id]), atomic_ids]))


@bp.route('/1.0/segment/<root_id>/subgraph', methods=['POST', 'GET'])
def handle_subgraph(root_id):
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    # Call ChunkedGraph
    cg = app_utils.get_cg()
    atomic_edges = cg.get_subgraph(int(root_id),
                                   get_edges=True,
                                   bounding_box=bounding_box,
                                   bb_is_coordinate=True)[0]
    # Return binary
    return app_utils.tobinary(atomic_edges)
