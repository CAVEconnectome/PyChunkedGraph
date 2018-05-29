from flask import Flask, jsonify, Response, request, make_response
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
from google.cloud import pubsub_v1
import json
import os
import numpy as np
import time

import chunkedgraph

HOME = os.path.expanduser("~")

app = Flask(__name__)
CORS(app)


def tobinary(ids):
    """ Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()


# @app.after_request
# def apply_caching(resp):
#     resp.headers['Access-Control-Allow-Origin'] = '*'
#     resp.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With," \
#                                                    " Content-Type, Accept"
#     resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
#     return resp


@app.route
def home():
    resp = make_response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With," \
                                                   " Content-Type, Accept"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.route('/1.0/graph/root', methods=['POST', 'GET'])
def handle_root():
    atomic_id = int(json.loads(request.data)[0])

    time_start = time.time()

    # Call ChunkedGraph
    root_id = cg.get_root(atomic_id, is_cg_id=True)

    dt = time.time() - time_start
    print("ROOT: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(root_id)


@app.route('/1.0/graph/merge', methods=['POST', 'GET'])
def handle_merge():
    node_1, node_2 = json.loads(request.data)

    time_start = time.time()

    # Call ChunkedGraph
    new_root = cg.add_edge([int(node_1[0]), int(node_2[0])], is_cg_id=True)

    dt = time.time() - time_start
    print("MERGE: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(new_root)


@app.route('/1.0/graph/split', methods=['POST', 'GET'])
def handle_split():
    data = json.loads(request.data)

    time_start = time.time()

    # Call ChunkedGraph
    new_roots = cg.remove_edges_mincut(int(data["sources"][0]),
                                       int(data["sinks"][0]),
                                       is_cg_id=True)

    dt = time.time() - time_start
    print("SPLIT: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(new_roots)


@app.route('/1.0/segment/<parent_id>/children', methods=['POST', 'GET'])
def handle_children(parent_id):
    # root_id = int(json.loads(request.data)[0])

    time_start = time.time()

    # Call ChunkedGraph
    # try:
    #     atomic_ids = cg.get_children(int(parent_id))
    # except:
    #     atomic_ids = np.array([])

    atomic_ids = cg.get_subgraph(int(parent_id), return_rg_ids=False)

    dt = time.time() - time_start
    print("CHILDREN: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(atomic_ids)


@app.route('/1.0/segment/<root_id>/leaves', methods=['POST', 'GET'])
def handle_leaves(root_id):

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None
    # root_id = int(json.loads(request.data)[0])

    time_start = time.time()

    # Call ChunkedGraph
    atomic_ids = cg.get_subgraph(int(root_id), return_rg_ids=False,
                                 bounding_box=bounding_box,
                                 bb_is_coordinate=True)

    dt = time.time() - time_start
    print("LEAVES: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(atomic_ids)


@app.route('/1.0/segment/<atomic_id>/leaves_from_leave', methods=['POST', 'GET'])
def handle_leaves_from_leave(atomic_id):
    # root_id = int(json.loads(request.data)[0])

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    time_start = time.time()

    # Call ChunkedGraph
    root_id = cg.get_root(int(atomic_id), is_cg_id=True)

    atomic_ids = cg.get_subgraph(root_id, return_rg_ids=False,
                                 bounding_box=bounding_box,
                                 bb_is_coordinate=True)

    dt = time.time() - time_start
    print("LEAVES FROM LEAVES: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(np.concatenate([np.array([root_id]), atomic_ids]))


@app.route('/1.0/segment/<root_id>/subgraph', methods=['POST', 'GET'])
def handle_subgraph(root_id):
    # root_id = int(json.loads(request.data)[0])

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    time_start = time.time()

    # Call ChunkedGraph
    atomic_edges = cg.get_subgraph(int(root_id),
                                   return_rg_ids=False,
                                   get_edges=True,
                                   bounding_box=bounding_box,
                                   bb_is_coordinate=True)[0]

    dt = time.time() - time_start
    print("SUBGRAPH: %3fms" % (dt * 1000))

    # Return binary
    return tobinary(atomic_edges)


if __name__ == '__main__':
    # Initialize chunkedgraph:
    cg = chunkedgraph.ChunkedGraph(table_id="basil")

    # Initialize google pubsub publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('neuromancer-seung-import',
                                      'pychunkedgraph')

    WSGIRequestHandler.protocol_version = "HTTP/1.1"

    app.run(host='0.0.0.0',
            port=4000,
            debug=True,
            threaded=True,
            ssl_context=(HOME + '/keys/server.crt',
                         HOME + '/keys/server.key'))
