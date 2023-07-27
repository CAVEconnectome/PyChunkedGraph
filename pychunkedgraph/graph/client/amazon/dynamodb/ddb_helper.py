import math
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer, Binary

from .timestamped_cell import TimeStampedCell
from .... import attributes
from ....utils.serializers import pad_node_id, serialize_key, serialize_uint64, deserialize_uint64


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
                
                print(f"----------row_value === {row_value}")
                
                for timestamp, column_value in row_value:
                    # find _Attribute for ddb_clm:qualifier and put cell & timestamp pair associated with it
                    row[attr].append(
                        # TimeStampedCell(
                        #     column_value.value
                        #     if isinstance(column_value, Binary)
                        #     else column_value,
                        #     int(timestamp),
                        # )
                        TimeStampedCell(
                            column_value,
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
