from flask import g, current_app
from google.auth import credentials, default as default_creds
from google.cloud import bigtable
from google.cloud import datastore

import numpy as np

# Hack the imports for now
from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.logging import flask_log_db

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


def get_datastore_client(config):
    project_id = config.get('project_id', 'pychunkedgraph')

    if config.get('emulate', False):
        credentials = DoNothingCreds()
    else:
        credentials, project_id = default_creds()

    client = datastore.Client(project=project_id, credentials=credentials)
    return client


def get_cg():
    if 'cg' not in cache:
        table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
        client = get_bigtable_client(current_app.config)
        cache["cg"] = chunkedgraph.ChunkedGraph(table_id=table_id,
                                                client=client)

    return cache["cg"]


def get_log_db():
    if 'log_db' not in cache:
        table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
        client = get_datastore_client(current_app.config)
        cache["log_db"] = flask_log_db.FlaskLogDatabase(table_id,
                                                        client=client)

    return cache["log_db"]


def tobinary(ids):
    """ Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()
