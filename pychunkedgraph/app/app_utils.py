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
    data1 = {
        "sources": [["91356497812401200", 815562.1875, 884530.4375, 859720]],
        "sinks": [["91356497812399839", 816066.8125, 884512.25, 859720]],
    }
    data2 = {
        "sources": [["91356497812394262", 815458.9375, 884707.6875, 859720]],
        "sinks": [["91356497812401228", 816143.625, 884547.3125, 859720]],
    }
    data3 = {
        "sources": [["95302094482834765", 930180.9375, 1031548.5625, 595360]],
        "sinks": [["95302094482834732", 930355.1875, 1031461.4375, 595360]],
    }
    data = data3

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

    # cross
    edges = np.array([[94879882017803318, 94809513273628034]],dtype=np.uint64)
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
        # foo_split(cache[table_id])
        foo_merge(cache[table_id])
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
