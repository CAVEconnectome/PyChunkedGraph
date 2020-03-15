import logging
import sys
import time

import numpy as np
from flask import current_app, json
from google.auth import credentials
from google.auth import default as default_creds
from google.cloud import bigtable, datastore

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.logging import flask_log_db, jsonformatter

CACHE = {}


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass


def jsonify_with_kwargs(data, **kwargs):
    kwargs.setdefault("separators", (",", ":"))

    if current_app.config["JSONIFY_PRETTYPRINT_REGULAR"] or current_app.debug:
        kwargs["indent"] = 2
        kwargs["separators"] = (", ", ": ")

    return current_app.response_class(
        json.dumps(data, **kwargs) + "\n", mimetype=current_app.config["JSONIFY_MIMETYPE"]
    )


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
    assert (table_id.startswith("pinky100_sv") or table_id.startswith("pinky100_neo"))
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
        CACHE["log_db"] = flask_log_db.FlaskLogDatabase(table_id, client=client,
                                                        credentials=credentials)

    return CACHE["log_db"]


def toboolean(value):
    """ Transform value to boolean type.
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
