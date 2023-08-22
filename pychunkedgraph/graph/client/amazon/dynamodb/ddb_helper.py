import math
from datetime import datetime
from typing import Dict, Iterable, Union, Any, Optional

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer, Binary

from .timestamped_cell import TimeStampedCell
from .... import attributes
from ....attributes import _Attribute
from ....utils.serializers import pad_node_id

MAX_DDB_BATCH_WRITE = 25
TIME_SLOT_INDEX = 0
VALUE_SLOT_INDEX = 1

LOCK_TIMESTAMP_COL_SUFFIX = '.ts'


# utility function to get DynamoDB attribute name (column name)
# from the given column object
def to_column_name(lock_column: _Attribute):
    return f"{lock_column.family_id}.{lock_column.key.decode()}"


# utility function to get DynamoDB attribute name (column name) holding
# the timestamp when the lock was acquired from the given lock column object
def to_lock_timestamp_column_name(lock_column: _Attribute):
    return f"{to_column_name(lock_column)}{LOCK_TIMESTAMP_COL_SUFFIX}"


class DdbHelper:
    
    def __init__(self):
        self._ddb_serializer = TypeSerializer()
        self._ddb_deserializer = TypeDeserializer()
        
        # TODO: ATTENTION
        # Fixed split between PK and SK may not be right approach
        # There are multiple key families with different sub-structure
        # Pending work:
        #  * Key sub-structure should be investigated
        #  * Key-range queries should be investigated
        #  * Split between PK and SK should be revisited to make sure that
        #    such key-range queries do _never_ run across different PKs or
        #    there should be some mechanism to make it working with multiple PKs
        # I'll put a hardcoded value of 18 to match ingestion default for now
        self._pk_key_shift = 18
        # TODO: ATTENTION
        # If number of bits to shift (or in other words split width) is variadic
        # which is implied by the key sub-structure and key-range queries,
        # mask and format should be calculated on the fly
        # and the same should be done in the ingestion script
        self._sk_key_mask = (1 << self._pk_key_shift) - 1
        pk_digits = math.ceil(math.log10(pow(2, 64 - self._pk_key_shift)))
        self._pk_int_format = f"0{pk_digits + 1}"
    
    @staticmethod
    def attribs_to_cells(
        attribs: Dict[_Attribute, Any],
        time_stamp: Optional[datetime] = None,
    ) -> dict[str, Iterable[TimeStampedCell]]:
        cells = {}
        for attrib_column, value in attribs.items():
            attr = attributes.from_key(attrib_column.family_id, attrib_column.key)
            if attr not in cells:
                cells[attr] = []
            cells[attr].append(
                TimeStampedCell(
                    value,
                    time_stamp.microsecond if time_stamp is not None else int(
                        TimeStampedCell.get_current_time_microseconds()),
                )
            )
        return cells
    
    def ddb_item_to_row(self, item, needs_deserialization: bool = False):
        row = {}
        pk = None
        sk = 0
        
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
            ddb_row_value = item[ddb_clm]
            row_value = self._ddb_deserializer.deserialize(ddb_row_value) if needs_deserialization else ddb_row_value
            if ddb_clm == "@":
                # for '@' row_value is int
                # TODO: store row version for optimistic locking (subject TBD)
                ver = row_value
            elif ddb_clm == "key":
                # for key row_value is string
                pk = row_value
            elif ddb_clm == "sk":
                # for sk row_value is int
                sk = int(row_value)
            else:
                # ddb_clm here is column_family.column_qualifier
                column_family, qualifier = ddb_clm.split(".")
                attr = attributes.from_key(column_family, qualifier.encode())
                
                if attr not in row:
                    row[attr] = []
                
                if attr in [attributes.Concurrency.Lock, attributes.Concurrency.IndefiniteLock]:
                    column_value = row_value
                    
                    ddb_timestamp_value = item.get(f"{ddb_clm}{LOCK_TIMESTAMP_COL_SUFFIX}", None)
                    
                    timestamp = self._ddb_deserializer.deserialize(
                        ddb_timestamp_value) if needs_deserialization else ddb_timestamp_value
                    
                    row[attr].append(
                        TimeStampedCell(
                            column_value,
                            int(timestamp)
                        )
                    )
                else:
                    for timestamp, column_value in row_value:
                        row[attr].append(
                            TimeStampedCell(
                                attr.deserialize(
                                    bytes(column_value)
                                    if isinstance(column_value, Binary)
                                    else column_value
                                ),
                                int(timestamp),
                            )
                        )
        
        b_real_key = self.to_real_key(pk, sk)
        
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
            
            if ddb_column not in columns:
                columns[ddb_column] = []
            
            for cell in cells_array:
                timestamp = cell.timestamp_int
                value = cell.value
                columns[ddb_column].append([
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
    
    def to_real_key(self, pk, sk):
        ikey = sk
        if pk[0].isdigit():
            real_key = pad_node_id(ikey)
        elif pk[0] in ["i", "f"]:
            real_key = f"{pk[0]}{pad_node_id(ikey)}"
        else:
            real_key = pk
        
        b_real_key = real_key.encode()
        return b_real_key
    
    def to_pk_sk(self, key: bytes):
        prefix, ikey, suffix = self._to_key_parts(key)
        if ikey is not None:
            pk = self._int_key_to_pk(ikey, prefix)
            sk = ikey
        else:
            pk = key.decode()
            sk = 0
        
        if suffix is not None and suffix.isnumeric():
            sk += int(suffix)
        
        return pk, sk
    
    def to_sk_range(
        self,
        start_key: bytes,
        end_key: bytes,
        start_inclusive: bool = True,
        end_inclusive: bool = True
    ):
        pk_start, sk_start = self.to_pk_sk(start_key)
        pk_end, sk_end = self.to_pk_sk(end_key)
        if pk_start is not None and pk_end is not None and pk_start != pk_end:
            raise ValueError("DynamoDB does not support range queries across different partition keys")
        
        if sk_start is not None:
            if not start_inclusive:
                sk_start = sk_start + 1
        #
        if sk_end is not None:
            if not end_inclusive:
                sk_end = sk_end - 1
        
        return pk_start if pk_start is not None else pk_end, sk_start, sk_end
    
    # def _to_key_parts(self, key: bytes):
    #     str_key = key.decode()
    #     prefix = None
    #     ikey = None
    #     if str_key[0].isdigit():
    #         return prefix, int(str_key)
    #     elif str_key[0] in ["f", "i"]:
    #         prefix = str_key[0]
    #         rest_of_the_key = str_key[1:]
    #         if rest_of_the_key.isnumeric():
    #             ikey = int(rest_of_the_key)
    #         return prefix, ikey
    #     else:
    #         return prefix, ikey
    
    def _to_key_parts(self, key: bytes):
        '''
        # A utility method to split the given key into prefix, an integer key, and a suffix
        #
        # The given key may be in any one of the following formats
        # 1. A padded 20-digit number
        #       E.g., 00076845692567897775
        # 2. A padded 19-digit number with "i" or "f" prefix (total 20 chars)
        #       E.g., f00144821212986474496 or i00145242668664881152
        # 3. A padded 19-digit number with "i" or "f" prefix and a suffix number separated by underscore
        #       E.g., i00216172782113783808_237
        # 4. Key for operations
        #       E.g., ioperations
        # 5. Key for meta
        #       E.g., meta
        #
        :param key:
        :return:
        '''
        str_key = key.decode()
        
        suffix = None
        key_without_suffix = str_key
        if "_" in str_key:
            parts = str_key.split("_")
            suffix = parts[-1]
            key_without_suffix = parts[0]
        
        prefix = None
        ikey = None
        
        if key_without_suffix[0].isdigit():
            return prefix, int(key_without_suffix), suffix
        elif key_without_suffix[0] in ["f", "i"]:
            prefix = key_without_suffix[0]
            rest_of_the_key = key_without_suffix[1:]
            if rest_of_the_key.isnumeric():
                ikey = int(rest_of_the_key)
            return prefix, ikey, suffix
        else:
            return prefix, ikey, suffix
    
    def _int_key_to_pk(self, ikey: int, prefix: str = None):
        return f"{'' if prefix is None else prefix}{(ikey >> self._pk_key_shift):{self._pk_int_format}}"
