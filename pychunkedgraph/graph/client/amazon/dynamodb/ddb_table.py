import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

from .ddb_helper import DdbHelper


class Table:
    def __init__(
            self,
            main_db,
            table_name,
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
        self._ddb_helper = DdbHelper()
    
    def read_rows(self):
        # ret = self._main_db.scan(
        #     TableName=self._table_name,
        #     Limit=self._row_page_size,
        # )
        ret = self._ddb_table.scan(Limit=self._row_page_size)
        items = ret.get("Items", [])
        
        rows = {}
        for item in items:
            print(f"Translating item to row ===== {item}")
            b_real_key, row = self._ddb_helper.ddb_item_to_row(item)
            rows[b_real_key] = row
        
        return TableRows(rows)


class TableRows:
    def __init__(self, rows):
        self._rows = rows
    
    def consume_all(self):
        pass
        
        # function to get value of _age
    
    def get_rows(self):
        return self._rows
    
    rows = property(get_rows)
