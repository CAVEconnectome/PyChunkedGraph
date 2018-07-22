from flask import g, current_app
from google.auth import credentials, default as default_creds
from google.cloud import bigtable

# Hack the imports for now
from pychunkedgraph.backend import chunkedgraph


class DoNothingCreds(credentials.Credentials):
    def refresh(self, request):
        pass


def get_client(config):
    project_id = config.get('project_id', 'pychunkedgraph')
    if config.get('emulate', False):
        credentials = DoNothingCreds()
    else:
        credentials, project_id = default_creds()
    client = bigtable.Client(admin=True,
                             project=project_id,
                             credentials=credentials)
    return client


def get_cg():
    if 'cg' not in g:
        table_id = current_app.config['CHUNKGRAPH_TABLE_ID']
        client = get_client(current_app.config)
        g.cg = chunkedgraph.ChunkedGraph(table_id=table_id,
                                         client=client)
    return g.cg
