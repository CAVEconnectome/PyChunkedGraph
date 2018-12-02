from flask import current_app
from google.auth import credentials, default as default_creds
from google.cloud import bigtable
from google.cloud import datastore

import sys
import numpy as np
import logging
import time

from pychunkedgraph.backend import chunkedgraph_meta
from pychunkedgraph.logging import flask_log_db, jsonformatter

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


# def get_datastore_client(config):
#     project_id = config.get('project_id', 'pychunkedgraph')
#
#     if config.get('emulate', False):
#         credentials = DoNothingCreds()
#     else:
#         credentials, project_id = default_creds()
#
#     client = datastore.Client(project=project_id, credentials=credentials)
#     return client


def get_mcg():
    if 'mcg' not in cache:
        instance_id = current_app.config['CHUNKGRAPH_INSTANCE_ID']
        table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
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
        cache["mcg"] = chunkedgraph_meta.ChunkedGraphMeta(
            instance_id=instance_id, client=client, logger=logger)
        return cache["mcg"]


# def get_log_db():
#     if 'log_db' not in cache:
#         table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
#         client = get_datastore_client(current_app.config)
#         cache["log_db"] = flask_log_db.FlaskLogDatabase(table_id,
#                                                         client=client)
#
#     return cache["log_db"]


def tobinary(ids):
    """ Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()
