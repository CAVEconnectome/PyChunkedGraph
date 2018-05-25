from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from google.cloud import pubsub_v1
import json
import os
import numpy as np

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


@app.route('/1.0/graph/root/', methods=['POST', 'GET'])
def handle_root():
    atomic_id = int(json.loads(request.data)[0])

    # Call ChunkedGraph
    root_id = cg.get_root(atomic_id, is_cg_id=True)

    # Return binary
    return tobinary(root_id)


@app.route('/1.0/graph/merge/', methods=['POST', 'GET'])
def handle_merge():
    node_1, node_2 = json.loads(request.data)

    # Call ChunkedGraph
    new_root = cg.add_edge([int(node_1[0]), int(node_2[0])], is_cg_id=True)

    # Return binary
    return tobinary(new_root)


@app.route('/1.0/graph/split/', methods=['POST', 'GET'])
def handle_split():
    node_1, node_2 = json.loads(request.data)

    # Call ChunkedGraph
    new_roots = cg.remove_edge([[int(node_1[0]), int(node_2[0])]],
                               is_cg_id=True)

    # Return binary
    return tobinary(new_roots)


@app.route('/1.0/segment/<root_id>/leaves/', methods=['POST', 'GET'])
def handle_leaves(root_id):
    # root_id = int(json.loads(request.data)[0])

    # Call ChunkedGraph
    atomic_ids = cg.get_subgraph(int(root_id), return_rg_ids=False)

    # Return binary
    return tobinary(atomic_ids)


if __name__ == '__main__':
    # Initialize chunkedgraph:
    cg = chunkedgraph.ChunkedGraph(dev_mode=False)

    # Initialize google pubsub publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('neuromancer-seung-import', 'pychunkedgraph')

    app.run(host='0.0.0.0',
            port=4000,
            debug=True,
            threaded=True,
            ssl_context=(HOME + '/keys/server.crt',
                         HOME + '/keys/server.key'))
