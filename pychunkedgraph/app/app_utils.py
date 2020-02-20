from flask import current_app
from google.auth import credentials, default as default_creds
from google.cloud import bigtable, datastore

import sys
import numpy as np
import logging
import time
import redis
import functools

from pychunkedgraph.logging import jsonformatter, flask_log_db
from pychunkedgraph.graph import chunkedgraph
from ..graph.utils.context_managers import TimeIt

cache = {}


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass


def get_bigtable_client(config):
    project_id = config.get("project_id", "pychunkedgraph")

    if config.get("emulate", False):
        credentials = DoNothingCreds()
    else:
        credentials, project_id = default_creds()

    client = bigtable.Client(admin=True, project=project_id, credentials=credentials)
    return client


def get_datastore_client(config):
    project_id = config.get("project_id", "pychunkedgraph")

    if config.get("emulate", False):
        credentials = DoNothingCreds()
    else:
        credentials, project_id = default_creds()

    client = datastore.Client(project=project_id, credentials=credentials)
    return client


def foo_split(cg):
    # l2 fail
    d1 = {
        "sources": [["87979484480978348", 717418.8125, 905186.875, 602200]],
        "sinks": [["87979553200440097", 717687.6875, 905449.25, 602200]],
    }
    # l5
    d2 = {
        "sources": [["88399772800622120", 729368.5, 847633.8125, 599680]],
        "sinks": [["88329404056440986", 728994.5, 847396.625, 599680]],
    }
    data = d1

    from collections import defaultdict

    data_dict = {}
    for k in ["sources", "sinks"]:
        data_dict[k] = defaultdict(list)
        for node in data[k]:
            node_id = node[0]
            x, y, z = node[1:]
            coordinate = np.array([x, y, z]) / cg.meta._ws_cv.resolution

            atomic_id = cg.get_atomic_id_from_coord(
                coordinate[0],
                coordinate[1],
                coordinate[2],
                parent_id=np.uint64(node_id),
            )

            data_dict[k]["id"].append(atomic_id)
            data_dict[k]["coord"].append(coordinate)

    from pychunkedgraph.graph.operation import MulticutOperation

    op = MulticutOperation(
        cg,
        user_id="test",
        source_ids=data_dict["sources"]["id"],
        sink_ids=data_dict["sinks"]["id"],
        source_coords=data_dict["sources"]["coord"],
        sink_coords=data_dict["sinks"]["coord"],
        bbox_offset=(240, 240, 24),
    )

    print(op._apply(operation_id="", timestamp=None))


def foo_merge(cg):
    from pychunkedgraph.graph.edits import add_edges

    # between 2
    edges = np.array([[94524946524577880, 94595315268752176]], dtype=np.uint64)
    with TimeIt("add_edges"):
        print(add_edges(cg, atomic_edges=edges))


def get_cg(table_id):
    # assert table_id.startswith("fly") or table_id.startswith("golden") or \
    #        table_id.startswith("pinky100_rv")

    if table_id not in cache:
        instance_id = current_app.config["CHUNKGRAPH_INSTANCE_ID"]

        # Create ChunkedGraph logging
        logger = logging.getLogger(f"{instance_id}/{table_id}")
        logger.setLevel(current_app.config["LOGGING_LEVEL"])

        # prevent duplicate logs from Flasks(?) parent logger
        logger.propagate = False

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(current_app.config["LOGGING_LEVEL"])
        formatter = jsonformatter.JsonFormatter(
            fmt=current_app.config["LOGGING_FORMAT"],
            datefmt=current_app.config["LOGGING_DATEFORMAT"],
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Create ChunkedGraph
        cache[table_id] = chunkedgraph.ChunkedGraph(graph_id=table_id)
        foo_split(cache[table_id])
        # foo_merge(cache[table_id])
    current_app.table_id = table_id
    return cache[table_id]


def get_log_db(table_id):
    if "log_db" not in cache:
        client = get_datastore_client(current_app.config)
        cache["log_db"] = flask_log_db.FlaskLogDatabase(table_id, client=client)
    return cache["log_db"]


def tobinary(ids):
    """ Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()


def tobinary_multiples(arr):
    """ Transform id(s) to binary format

    :param arr: list of uint64 or list of uint64s
    :return: binary
    """
    return [np.array(arr_i).tobytes() for arr_i in arr]
