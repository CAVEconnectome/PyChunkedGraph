import logging
import math
from datetime import datetime
from typing import Dict, Iterable, Union, Any, Optional

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer, Binary

from .timestamped_cell import TimeStampedCell
from .... import attributes
from ....utils.serializers import pad_node_id

MAX_DDB_BATCH_WRITE = 25
TIME_SLOT_INDEX = 0
VALUE_SLOT_INDEX = 1


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
            attribs: Dict[attributes._Attribute, Any],
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
    
    def raw_ddb_item_to_row(self, item):
        row = {}
        pk = None
        sk = 0
        
        # each item comes with 'key', 'sk', [column_family] and '@' columns
        item_keys = list(item.keys())
        item_keys.sort()
        # ddb_clm is one of 'key', 'sk', [column_family.column_qualifier], '@'
        for ddb_clm in item_keys:
            ddb_row_value = item[ddb_clm]
            row_value = self._ddb_deserializer.deserialize(ddb_row_value)
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
                
                for timestamp, column_value in row_value:
                    # find _Attribute for ddb_clm:qualifier and put cell & timestamp pair associated with it
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
        
        if pk[0].isdigit():
            ikey = (int(pk) << self._pk_key_shift) | sk
            real_key = pad_node_id(ikey)
        elif pk[0] in ["i", "f"]:
            ikey = (int(pk[1:]) << self._pk_key_shift) | sk
            real_key = pad_node_id(ikey)
        else:
            real_key = pk
        
        b_real_key = real_key.encode()
        return b_real_key, row
    
    def ddb_item_to_row(self, item):
        row = {}
        pk = None
        sk = 0
        
        # each item comes with 'key', 'sk', [column_family] and '@' columns
        item_keys = list(item.keys())
        item_keys.sort()
        # ddb_clm is one of 'key', 'sk', [column_family.column_qualifier], '@'
        for ddb_clm in item_keys:
            # ddb_row_value = item[ddb_clm]
            # row_value = self._ddb_deserializer.deserialize(ddb_row_value)
            row_value = item[ddb_clm]
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
                
                for timestamp, column_value in row_value:
                    # find _Attribute for ddb_clm:qualifier and put cell & timestamp pair associated with it
                    row[attr].append(
                        TimeStampedCell(
                            attr.deserialize(
                                bytes(column_value)
                                if isinstance(column_value, Binary)
                                else column_value
                            ),
                            int(timestamp),
                        )
                        # TimeStampedCell(
                        #     column_value,
                        #     int(timestamp),
                        # )
                    )
        
        if pk[0].isdigit():
            ikey = (int(pk) << self._pk_key_shift) | sk
            real_key = pad_node_id(ikey)
        elif pk[0] in ["i", "f"]:
            ikey = (int(pk[1:]) << self._pk_key_shift) | sk
            real_key = pad_node_id(ikey)
        else:
            real_key = pk
        
        b_real_key = real_key.encode()
        
        return b_real_key, row
    
    def row_to_ddb_item(
            self,
            row: dict[str, Union[bytes, dict[attributes._Attribute, Iterable[TimeStampedCell]]]]
    ) -> dict[str, Any]:
        pk, sk = self.to_pk_sk(row['key'])
        item = {'key': pk, 'sk': sk}
        
        columns = {}
        for attrib_column, cells_array in row.items():
            if not isinstance(attrib_column, attributes._Attribute):
                continue
            
            family = attrib_column.family_id
            qualifier = attrib_column.key.decode()
            
            # form column names for DDB like 0.parent, 0.children etc
            ddb_column = f"{family}.{qualifier}"
            if ddb_column not in columns:
                columns[ddb_column] = []
            
            for cell in cells_array:
                timestamp = cell.timestamp
                value = cell.value
                columns[ddb_column].append([
                    timestamp,  # timestamp is at TIME_SLOT_INDEX position
                    attrib_column.serializer.serialize(value),  # cell value is at VALUE_SLOT_INDEX position
                    # value,  # cell value is at VALUE_SLOT_INDEX position
                ])
            
            # sort so the latest timestamp would always be at 0 index
            for value_list in columns.values():
                value_list.sort(key=lambda it: it[TIME_SLOT_INDEX], reverse=True)
        
        for k, v in columns.items():
            item[k] = v
        
        return item
    
    def to_pk_sk(self, key: bytes):
        skey = key.decode()
        if skey[0].isdigit():
            ikey = int(skey)
            pk = f"{(ikey >> self._pk_key_shift):{self._pk_int_format}}"
            sk = ikey & self._sk_key_mask
        elif skey[0] in ["f", "i"]:
            # TODO: keys with "i" prefix may have weird suffix, like _05 and couldn't be converted to int like below
            ikey = int(skey[1:])
            pk = f"{(ikey >> self._pk_key_shift):{self._pk_int_format}}"
            sk = ikey & self._sk_key_mask
        else:
            pk = skey
            sk = 0
        return pk, sk
