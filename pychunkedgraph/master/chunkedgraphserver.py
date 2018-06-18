from flask import Flask, jsonify, Response, request, make_response, g
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
from google.cloud import pubsub_v1
import json
import os
import numpy as np
import sys
import time
import datetime
# import pymongo
import logging
import config

# Hack the imports for now
sys.path.append("..")
from pychunkedgraph.backend import chunkedgraph

HOME = os.path.expanduser("~")

app = Flask(__name__)
# Tracking(app)
CORS(app)

# mongo_client = pymongo.MongoClient('localhost:2017/')
# mongo_db = mongo_client.mongo_db
# logs = mongo_db.logs


# --------------
# ------ Helpers
# --------------

def tobinary(ids):
    """ Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()


# -------------------------------
# ------ Access control and index
# -------------------------------

@app.route('/')
@app.route("/index")
def index():
    return "PyChunkedGraph Server"


@app.route
def home():
    resp = make_response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With," \
                                                   " Content-Type, Accept"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


# -------------------------------
# ------ Measurements and Logging
# -------------------------------

@app.before_request
def before_request():
    print("NEW REQUEST:", datetime.datetime.now(), request.url)
    g.request_start_time = time.time()


@app.after_request
def after_request(response):
    dt = (time.time() - g.request_start_time) * 1000

    url_split = request.url.split("/")
    app.logger.info("%s - %s - %s - %s - %f.3" %
                    (request.path.split("/")[-1], "1",
                     "".join([url_split[-2], "/", url_split[-1]]),
                     str(request.data), dt))

    print("Response time: %.3fms" % (dt))
    return response


@app.errorhandler(500)
def internal_server_error(error):
    dt = (time.time() - g.request_start_time) * 1000

    url_split = request.url.split("/")
    app.logger.error("%s - %s - %s - %s - %f.3" %
                     (request.path.split("/")[-1],
                      "Server Error: " + error,
                      "".join([url_split[-2], "/", url_split[-1]]),
                      str(request.data), dt))
    return 500


@app.errorhandler(Exception)
def unhandled_exception(e):
    dt = (time.time() - g.request_start_time) * 1000

    url_split = request.url.split("/")
    app.logger.error("%s - %s - %s - %s - %f.3" %
                     (request.path.split("/")[-1],
                      "Exception: " + str(e),
                      "".join([url_split[-2], "/", url_split[-1]]),
                      str(request.data), dt))
    return 500

# -------------------
# ------ Applications
# -------------------


@app.route('/1.0/graph/root', methods=['POST', 'GET'])
def handle_root():
    atomic_id = int(json.loads(request.data)[0])

    # Call ChunkedGraph
    root_id = cg.get_root(atomic_id, is_cg_id=True)

    # Return binary
    return tobinary(root_id)


@app.route('/1.0/graph/merge', methods=['POST', 'GET'])
def handle_merge():
    node_1, node_2 = json.loads(request.data)\

    thread_id = np.random.randint(0, 2**52)

    # Call ChunkedGraph
    new_root = cg.add_edge_locked(thread_id, [int(node_1[0]), int(node_2[0])])

    # Return binary
    return tobinary(new_root)


@app.route('/1.0/graph/split', methods=['POST', 'GET'])
def handle_split():
    data = json.loads(request.data)

    thread_id = np.random.randint(0, 2**52)

    # Call ChunkedGraph
    new_roots = cg.remove_edges_mincut_locked(thread_id,
                                              int(data["sources"][0][0]),
                                              int(data["sinks"][0][0]),
                                              data["sources"][0][1:],
                                              data["sinks"][0][1:])
    # Return binary
    return tobinary(new_roots)


@app.route('/1.0/segment/<parent_id>/children', methods=['POST', 'GET'])
def handle_children(parent_id):
    # Call ChunkedGraph
    if False or cg.get_chunk_id_from_node_id(parent_id)[0] > 2:
        print("MIP1 meshes")
        try:
            # atomic_ids = cg.get_children(int(parent_id))
            atomic_ids = cg.get_subgraph(int(parent_id), return_rg_ids=False,
                                         stop_lvl=2)
        except:
            atomic_ids = np.array([])
    else:
        print("MIP0 meshes")
        try:
            atomic_ids = cg.get_subgraph(int(parent_id), return_rg_ids=False)
        except:
            atomic_ids = np.array([])

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

    # Call ChunkedGraph
    atomic_ids = cg.get_subgraph(int(root_id), return_rg_ids=False,
                                 bounding_box=bounding_box,
                                 bb_is_coordinate=True)

    # print(atomic_ids)

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

    # Call ChunkedGraph
    root_id = cg.get_root(int(atomic_id), is_cg_id=True)

    atomic_ids = cg.get_subgraph(root_id, return_rg_ids=False,
                                 bounding_box=bounding_box,
                                 bb_is_coordinate=True)
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

    # Call ChunkedGraph
    atomic_edges = cg.get_subgraph(int(root_id),
                                   return_rg_ids=False,
                                   get_edges=True,
                                   bounding_box=bounding_box,
                                   bb_is_coordinate=True)[0]
    # Return binary
    return tobinary(atomic_edges)


def configure_app(app):
    # config_name = os.getenv('FLASK_CONFIGURATION', 'default')
    # app.config.from_object(config[config_name])

    # Load logging scheme from config.py
    app.config.from_object(config.BaseConfig)

    # Configure logging
    handler = logging.FileHandler(app.config['LOGGING_LOCATION'])
    handler.setLevel(app.config['LOGGING_LEVEL'])
    formatter = logging.Formatter(app.config['LOGGING_FORMAT'])
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)


if __name__ == '__main__':
    assert len(sys.argv) == 3

    table_id = sys.argv[1]
    port = int(sys.argv[2])

    # Initialize chunkedgraph:
    cg = chunkedgraph.ChunkedGraph(table_id=table_id)

    # Initialize google pubsub publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('neuromancer-seung-import',
                                      'pychunkedgraph')

    # Set HTTP protocol
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    # WSGIRequestHandler.protocol_version = "HTTP/2.0"

    configure_app(app)

    print("Table: %s; Port: %d; Log-Path: %s" %
          (table_id, port, app.config['LOGGING_LOCATION']))

    app.run(host='0.0.0.0',
            port=port,
            debug=True,
            threaded=True,
            ssl_context=(HOME + '/keys/server.crt',
                         HOME + '/keys/server.key'))
