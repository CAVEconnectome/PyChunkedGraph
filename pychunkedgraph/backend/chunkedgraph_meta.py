from google.cloud import bigtable

from pychunkedgraph.backend import chunkedgraph

class ChunkedGraphMeta(object):
    """ Manages multiple pychunkedgraphs """

    def __init__(self, client=None, instance_id='pychunkedgraph',
                 project_id="neuromancer-seung-import", credentials=None,
                 logger=None):

        if client is not None:
            self._client = client
        else:
            self._client = bigtable.Client(project=project_id, admin=True,
                                           credentials=credentials)

        self._instance = self.client.instance(instance_id)
        self._logger = logger

        self._loaded_tables = {}

    @property
    def client(self):
        return self._client

    @property
    def instance(self):
        return self._instance

    @property
    def instance_id(self):
        return self.instance.instance_id

    @property
    def logger(self):
        return self._logger

    @property
    def project_id(self):
        return self.client.project

    def get_existing_tables(self):
        """ Collects table_ids of existing tables

        Annotation tables start with `anno`

        :return: list
        """
        tables = self.instance.list_tables()

        graph_tables = []
        for table in tables:
            table_name = table.name.split("/")[-1]
            if not table_name.startswith("anno"):
                graph_tables.append(table_name)

        return graph_tables

    def _is_loaded(self, table_id):

        """ Checks whether table_id is in _loaded_tables

        :param table_id: str
        :return: bool
        """
        return table_id in self._loaded_tables

    def _load_table(self, table_id):
        """ Loads existing table

        :param table_id: str
        :return: bool
            success
        """

        if self._is_loaded(table_id):
            return True

        if table_id in self.get_existing_tables():
            self._loaded_tables[table_id] = \
                chunkedgraph.ChunkedGraph(table_id=table_id, client=self.client,
                                          logger=self.logger,
                                          instance_id=self.instance_id)
            return True
        else:
            print("Table id does not exist")
            return False

    def get_chunkedgraph(self, table_id):
        if self._load_table(table_id):
            return self._loaded_tables[table_id]
        else:
            return None