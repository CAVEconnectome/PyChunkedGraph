from datetime import datetime
from typing import Dict, Iterable, Union, Any, Optional

from boto3.dynamodb.types import TypeDeserializer, Binary

from .key_translator import KeyTranslator
from .timestamped_cell import TimeStampedCell
from .utils import append, get_current_time_microseconds, to_microseconds
from .... import attributes
from ....attributes import _Attribute

MAX_DDB_BATCH_WRITE = 25
TIME_SLOT_INDEX = 0
VALUE_SLOT_INDEX = 1

LOCK_TIMESTAMP_COL_SUFFIX = '.ts'


# utility function to get DynamoDB attribute name (column name)
# from the given column object
def to_column_name(column: _Attribute):
    return f"{column.family_id}.{column.key.decode()}"


# utility function to get DynamoDB attribute name (column name) holding
# the timestamp when the lock was acquired from the given lock column object
def to_lock_timestamp_column_name(lock_column: _Attribute):
    return f"{to_column_name(lock_column)}{LOCK_TIMESTAMP_COL_SUFFIX}"


class DdbTranslator:
    """
    Translator class that provides a set of methods to translate between the internal DynamoDB "item" format and the
    "row" and "cells" representation used by client code
    """
    
    def __init__(self):
        self._ddb_deserializer = TypeDeserializer()
        self._key_translator = KeyTranslator()
    
    def attribs_to_cells(
        self,
        attribs: Dict[_Attribute, Any],
        time_stamp: Optional[datetime] = None,
    ) -> dict[str, Iterable[TimeStampedCell]]:
        cells = {}
        for attrib_column, value in attribs.items():
            attr = attributes.from_key(attrib_column.family_id, attrib_column.key)
            append(cells, attr, TimeStampedCell(
                value,
                to_microseconds(time_stamp) if time_stamp else get_current_time_microseconds(),
            ))
        return cells
    
    def ddb_item_to_row(self, item):
        row = {}
        pk = None
        sk = ''
        
        # Item is a dict object retrieved from Amazon DynamoDB (DDB).
        # The dictionary object is keyed by column name (i.e., attribute name) in the DDB table.
        # The value for each column is an array and represents the column values history over time.
        # Each element in the array is also an array containing two elements [timestamp, column_value]
        # representing the value of the given column at a given time.
        #
        # Instead of the column value history, some columns may contain the value directly
        # E.g., the columns for Locks (i.e., "attributes.Concurrency.Lock" and "attributes.Concurrency.IndefiniteLock")
        # directly store the value in the column. For such columns, the column value history is not stored and the
        # timestamp when the column was added (i.e., when the lock was acquired) is stored in a separate column
        # with the ".ts" suffix.
        #
        item_keys = [k for k in item.keys() if not k.endswith(LOCK_TIMESTAMP_COL_SUFFIX)]
        item_keys.sort()
        
        # ddb_clm is one of the followings: 'key' (primary key), 'sk' (sort key), '@' (row version),
        # and other columns with the format [column_family.column_qualifier]
        for ddb_clm in item_keys:
            row_value = item[ddb_clm]
            if ddb_clm == "@":
                # for '@' row_value is int
                # TODO: store row version for optimistic locking (subject TBD)
                ver = row_value
            elif ddb_clm == "key":
                pk = row_value
            elif ddb_clm == "sk":
                sk = row_value
            else:
                # ddb_clm here is column_family.column_qualifier
                column_family, qualifier = ddb_clm.split(".")
                attr = attributes.from_key(column_family, qualifier.encode())
                
                if attr in [attributes.Concurrency.Lock, attributes.Concurrency.IndefiniteLock]:
                    column_value = row_value
                    
                    timestamp = item.get(f"{ddb_clm}{LOCK_TIMESTAMP_COL_SUFFIX}", None)
                    
                    append(row, attr, TimeStampedCell(
                        column_value,
                        int(timestamp)
                    ))
                
                else:
                    for timestamp, column_value in row_value:
                        if column_value:
                            append(row, attr, TimeStampedCell(
                                attr.deserialize(
                                    bytes(column_value)
                                    if isinstance(column_value, Binary)
                                    else column_value
                                ),
                                int(timestamp),
                            ))
        
        b_real_key = self._key_translator.to_unified_key(pk, sk)
        
        return b_real_key, row
    
    def row_to_ddb_item(
        self,
        row: dict[str, Union[bytes, dict[_Attribute, Iterable[TimeStampedCell]]]]
    ) -> dict[str, Any]:
        pk, sk = self.to_pk_sk(row['key'])
        item = {'key': pk, 'sk': sk}
        
        columns = {}
        for attrib_column, cells_array in row.items():
            if not isinstance(attrib_column, _Attribute):
                continue
            
            family = attrib_column.family_id
            qualifier = attrib_column.key.decode()
            # form column names for DDB like 0.parent, 0.children etc
            ddb_column = f"{family}.{qualifier}"
            
            if attrib_column in [attributes.Concurrency.Lock, attributes.Concurrency.IndefiniteLock]:
                # for Lock and IndefiniteLock, the column value history is not stored in DDB
                # instead, the timestamp when the lock was acquired is stored in a separate column
                # with the ".ts" suffix
                ddb_timestamp_column = f"{ddb_column}{LOCK_TIMESTAMP_COL_SUFFIX}"
                item[ddb_timestamp_column] = cells_array[0].timestamp_int
                item[ddb_column] = cells_array[0].value
                continue
            
            for cell in cells_array:
                timestamp = cell.timestamp_int
                value = cell.value
                append(columns, ddb_column, [
                    timestamp,  # timestamp is at TIME_SLOT_INDEX position
                    
                    # cell value is at VALUE_SLOT_INDEX position
                    bytes(value) if isinstance(value, Binary) else attrib_column.serializer.serialize(value),
                ])
            
            # sort so the latest timestamp would always be at 0 index
            for value_list in columns.values():
                value_list.sort(key=lambda it: it[TIME_SLOT_INDEX], reverse=True)
        
        for k, v in columns.items():
            item[k] = v
        
        return item
    
    def to_pk_sk(self, key: bytes):
        return self._key_translator.to_pk_sk(key)
    
    def to_sk_range(
        self,
        start_key: bytes,
        end_key: bytes,
        start_inclusive: bool = True,
        end_inclusive: bool = True
    ):
        return self._key_translator.to_sk_range(
            start_key,
            end_key,
            start_inclusive,
            end_inclusive,
        )
