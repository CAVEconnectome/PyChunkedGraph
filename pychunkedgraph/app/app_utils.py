from flask import current_app
from google.auth import credentials, default as default_creds
from google.cloud import bigtable

import sys
import numpy as np
import logging
import time

from pychunkedgraph.logging import jsonformatter
from pychunkedgraph.backend import chunkedgraph

cache = {}


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass


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


def get_cg(table_id):
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
    return cache[table_id]


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


def json_serialize_nd_array(arr):
    if not isinstance(arr, np.ndarray) and not isinstance(arr, list):
        return arr

    if len(arr) == 0:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        else:
            return arr

    if isinstance(arr[0], list) or isinstance(arr[0], np.ndarray):
        return [json_serialize_nd_array(arr[i]) for i in range(len(arr))]
    else:
        if isinstance(arr, np.ndarray):
            return arr.tolist()