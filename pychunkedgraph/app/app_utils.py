import logging
import sys
import time

import numpy as np
from flask import current_app, json
from google.auth import credentials
from google.auth import default as default_creds
from google.cloud import bigtable, datastore

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions
from pychunkedgraph.logging import flask_log_db, jsonformatter

import networkx as nx
from scipy import spatial


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

CACHE = {}


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass


def jsonify_with_kwargs(data, as_response=True, **kwargs):
    kwargs.setdefault("separators", (",", ":"))

    if current_app.config["JSONIFY_PRETTYPRINT_REGULAR"] or current_app.debug:
        kwargs["indent"] = 2
        kwargs["separators"] = (", ", ": ")

    resp = json.dumps(data, **kwargs)
    if as_response:
        return current_app.response_class(
            resp + "\n", mimetype=current_app.config["JSONIFY_MIMETYPE"]
        )
    else:
        return resp


def get_bigtable_client(config):
    project_id = config.get("PROJECT_ID", None)

    if config.get("emulate", False):
        credentials = DoNothingCreds()
    elif project_id is not None:
        credentials, _ = default_creds()
    else:
        credentials, project_id = default_creds()

    client = bigtable.Client(admin=True, project=project_id, credentials=credentials)
    return client


def get_datastore_client(config):
    project_id = config.get("PROJECT_ID", None)

    if config.get("emulate", False):
        credentials = DoNothingCreds()
    elif project_id is not None:
        credentials, _ = default_creds()
    else:
        credentials, project_id = default_creds()

    client = datastore.Client(project=project_id, credentials=credentials)
    return client


def get_cg(table_id):
    assert (
        table_id.startswith("fly")
        or table_id.startswith("golden")
        or table_id.startswith("pinky100_rv")
        or table_id.startswith("pinky100_arv")
    )

    if table_id not in CACHE:
        instance_id = current_app.config["CHUNKGRAPH_INSTANCE_ID"]
        client = get_bigtable_client(current_app.config)

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
        CACHE[table_id] = chunkedgraph.ChunkedGraph(
            table_id=table_id, instance_id=instance_id, client=client, logger=logger
        )

    current_app.table_id = table_id
    return CACHE[table_id]


def get_log_db(table_id):
    if "log_db" not in CACHE:
        client = get_datastore_client(current_app.config)
        CACHE["log_db"] = flask_log_db.FlaskLogDatabase(
            table_id, client=client, credentials=credentials
        )

    return CACHE["log_db"]


def toboolean(value):
    """Transform value to boolean type.
    :param value: bool/int/str
    :return: bool
    :raises: ValueError, if value is not boolean.
    """
    if not value:
        raise ValueError("Can't convert null to boolean")

    if isinstance(value, bool):
        return value
    try:
        value = value.lower()
    except:
        raise ValueError(f"Can't convert {value} to boolean")

    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False

    raise ValueError(f"Can't convert {value} to boolean")


def tobinary(ids):
    """Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()


def tobinary_multiples(arr):
    """Transform id(s) to binary format

    :param arr: list of uint64 or list of uint64s
    :return: binary
    """
    return [np.array(arr_i).tobytes() for arr_i in arr]


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
