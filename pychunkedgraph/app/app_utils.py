import logging
import sys
from time import gmtime

import numpy as np
from flask import current_app, json, request
from google.auth import credentials
from google.auth import default as default_creds
from google.cloud import bigtable, datastore

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.logging import flask_log_db, jsonformatter
from pychunkedgraph.graph import (
    exceptions as cg_exceptions,
)
from functools import wraps
from werkzeug.datastructures import ImmutableMultiDict
import time

CACHE = {}


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass


def remap_public(func=None, *, edit=False, check_node_ids=False):
    def mydecorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            virtual_tables = current_app.config.get("VIRTUAL_TABLES", {})
            table_id = kwargs.get("table_id")
            http_args = request.args.to_dict()

            if table_id is None:
                # then no table remapping necessary
                return f(*args, **kwargs)
            if not table_id in virtual_tables:
                # if table table_id isn't in virtual
                # tables then just return
                return f(*args, **kwargs)
            else:
                # then we have a virtual table
                if edit:
                    raise cg_exceptions.Unauthorized(
                        "No edits allowed on virtual tables"
                    )
                # then we want to remap the table name
                new_table = virtual_tables[table_id]["table_id"]
                kwargs["table_id"] = new_table
                v_timestamp = virtual_tables[table_id]["timestamp"]
                v_timetamp_float = time.mktime(v_timestamp.timetuple())

                # we want to fix timestamp parameters too
                http_args["timestamp"] = v_timetamp_float
                http_args["timestamp_future"] = v_timetamp_float
                request.args = ImmutableMultiDict(http_args)

                cg = get_cg(new_table)

                def assert_node_prop(prop):
                    node_id = kwargs.get(prop, None)
                    if node_id is not None:
                        node_id = int(node_id)
                        # check if this root_id is valid at this timestamp
                        timestamp = cg.get_node_timestamps([node_id])
                        if not np.all(timestamp < np.datetime64(v_timestamp)):
                            raise cg_exceptions.Unauthorized(
                                "root_id not valid at timestamp"
                            )

                assert_node_prop("root_id")
                assert_node_prop("node_id")

                if check_node_ids:
                    node_ids = np.array(
                        json.loads(request.data)["node_ids"], dtype=np.uint64
                    )
                    timestamps = cg.get_node_timestamps(node_ids)
                    if not np.all(timestamps < np.datetime64(v_timestamp)):
                        raise cg_exceptions.Unauthorized(
                            "node_ids are all not valid at timestamp"
                        )

                return f(*args, **kwargs)

        return decorated_function

    if func:
        return mydecorator(func)
    else:
        return mydecorator


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


def get_cg(table_id, skip_cache: bool = False):
    from pychunkedgraph.graph.client import get_default_client_info

    assert table_id in current_app.config["PCG_GRAPH_IDS"]

    current_app.table_id = table_id
    if skip_cache is False:
        try:
            return CACHE[table_id]
        except KeyError:
            pass

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
    formatter.converter = gmtime
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Create ChunkedGraph
    cg = ChunkedGraph(graph_id=table_id, client_info=get_default_client_info())
    if skip_cache is False:
        CACHE[table_id] = cg
    return cg


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
