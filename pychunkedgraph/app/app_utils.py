from flask import current_app, json
from google.auth import credentials, default as default_creds
from google.cloud import bigtable, datastore

import sys
import numpy as np
import logging
import time
import redis
import functools

from pychunkedgraph.logging import jsonformatter, flask_log_db
from pychunkedgraph.backend import chunkedgraph

cache = {}


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass

def jsonify_with_kwargs(data, **kwargs):
    kwargs.setdefault('separators', (",", ":"))

    if current_app.config["JSONIFY_PRETTYPRINT_REGULAR"] or current_app.debug:
        kwargs['indent'] = 2
        kwargs['separators'] = (", ", ": ")

    return current_app.response_class(
        json.dumps(data, **kwargs) + "\n",
        mimetype=current_app.config["JSONIFY_MIMETYPE"],
    )


def get_bigtable_client(config):
    project_id = config.get('project_id', 'pychunkedgraph')

    if config.get('emulate', False):
        credentials = DoNothingCreds()
    else:
        credentials, project_id = default_creds()

    client = bigtable.Client(admin=True,
                             project=project_id,
                             credentials=credentials)
    return client


def get_datastore_client(config):
    project_id = config.get('project_id', 'pychunkedgraph')

    if config.get('emulate', False):
        credentials = DoNothingCreds()
    else:
        credentials, project_id = default_creds()

    client = datastore.Client(project=project_id, credentials=credentials)
    return client


def get_cg(table_id):
    assert table_id.startswith("fly") or table_id.startswith("golden") or \
           table_id.startswith("pinky100_rv")

    if table_id not in cache:
        instance_id = current_app.config['CHUNKGRAPH_INSTANCE_ID']
        client = get_bigtable_client(current_app.config)

        # Create ChunkedGraph logging
        logger = logging.getLogger(f"{instance_id}/{table_id}")
        logger.setLevel(current_app.config['LOGGING_LEVEL'])

        # prevent duplicate logs from Flasks(?) parent logger
        logger.propagate = False

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(current_app.config['LOGGING_LEVEL'])
        formatter = jsonformatter.JsonFormatter(
            fmt=current_app.config['LOGGING_FORMAT'],
            datefmt=current_app.config['LOGGING_DATEFORMAT'])
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Create ChunkedGraph
        cache[table_id] = chunkedgraph.ChunkedGraph(table_id=table_id,
                                                    instance_id=instance_id,
                                                    client=client,
                                                    logger=logger)
    current_app.table_id = table_id
    return cache[table_id]


def get_log_db(table_id):
    if 'log_db' not in cache:
        client = get_datastore_client(current_app.config)
        cache["log_db"] = flask_log_db.FlaskLogDatabase(table_id,
                                                        client=client)

    return cache["log_db"]


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
