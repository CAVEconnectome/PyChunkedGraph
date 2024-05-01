# NOTE: ALL THE CLASSES IN THIS FILE ARE ONLY USED BY THE TEST CODE FOR INSPECTING THE ITEMS WRITTEN TO THE DB
# AND IS NOT MEANT TO BE USED BY THE ACTUAL CODE.

# The test code uses the "_table" internal variable of the pychunkedgraph client to inspect the items in the table.
# The test code assumes the "_table" to provide Google BitTable compatible APIs.
# The classes in this file provide the adapter that interacts with the Amazon DynamoDB table so that the test code can
# use the Google BitTable compatible APIs.

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

from pychunkedgraph.graph import attributes
from .item_compressor import ItemCompressor


class Table:
    """
    An adapter for an Amazon DynamoDB table.
    
    NOTE: THIS CLASS IS ONLY USED BY THE TEST CODE FOR INSPECTING THE ITEMS WRITTEN TO THE DB
    AND IS NOT MEANT TO BE USED BY THE ACTUAL CODE.
    """
    
    def __init__(
        self,
        main_db,
        table_name,
        translator,
        compressor: ItemCompressor,
        boto3_conf,
        **kwargs
    ):
        dynamodb = boto3.resource('dynamodb', config=boto3_conf, **kwargs)
        self._ddb_table = dynamodb.Table(table_name)
        self._main_db = main_db
        self._table_name = table_name
        self._row_page_size = 1000
        self._ddb_serializer = TypeSerializer()
        self._ddb_deserializer = TypeDeserializer()
        self._ddb_translator = translator
        self._ddb_item_compressor = compressor
    
    def read_rows(self):
        ret = self._ddb_table.scan(Limit=self._row_page_size)
        items = ret.get("Items", [])
        
        rows = {}
        for item in items:
            item = self._ddb_item_compressor.decompress(item)
            b_real_key, row = self._ddb_translator.ddb_item_to_row(item)
            rows[b_real_key] = Row(row)
        
        return TableRows(rows)
    
    @property
    def ddb_table(self):
        return self._ddb_table


class TableRows:
    def __init__(self, rows):
        self._rows = rows
    
    def consume_all(self):
        pass
    
    @property
    def rows(self):
        return self._rows
    
    @property
    def cells(self):
        return self._rows


class Row:
    def __init__(self, columns):
        __cells = {}
        for attr, value in columns.items():
            if isinstance(attr, attributes._Attribute):
                __cells[attr.family_id] = __cells.get(attr.family_id, {})
                __cells[attr.family_id][attr.key] = value
        self._cells = __cells
    
    @property
    def cells(self):
        return self._cells
