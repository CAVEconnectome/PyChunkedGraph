import bz2
import pickle
from typing import List, Dict

from boto3.dynamodb.types import Binary


class ItemCompressor:
    """
    A utility class to compress and decompress DynamoDB items. Compressing items before storing them in DynamoDB is
    beneficial for saving cost and improving performance, especially for large items.
    
    The class compresses all attributes of the given dictionary representing a DynamoDB item into a single attribute
    named "v". The key attributes (partition key and sort key) and the attributes specified in the "exclude_keys"
    are not compressed and are returned as-is.
    
    For example, given "_pk_name" = "pk" and "_sk_name" = "sk" and "_exclude_keys" = ["attrib1","attrib3"]
        
        ddb_item = {
            "pk":"pk1",
            "sk":"sk1",
            "attrib1":"value1",
            "attrib2":"value2",
            "attrib3":"value3",
            "attrib4":"value4",
            "attrib5":"value5",
        }
        compress(ddb_item)
        
    returns
        
        {
            "pk":"pk1",
            "sk":"sk1",
            "attrib1":"value1", # returned as-is since it's excluded from compression
            "attrib3":"value3", # returned as-is since it's excluded from compression
            "v":b'...compressed value...' # all other key/value pairs of the dict are compressed into a single key: "v"
        }
        
    passing the returned item to "decompress", returns the original dict
        
        i.e., decompress(compress(ddb_item)) returns the following
        {
            "pk":"pk1",
            "sk":"sk1",
            "attrib1":"value1",
            "attrib2":"value2",
            "attrib3":"value3",
            "attrib4":"value4",
            "attrib5":"value5",
        }
    """
    
    def __init__(
        self,
        pk_name: str,
        sk_name: str,
        exclude_keys: List[str],
    ):
        """
        :param pk_name: Name of the partition key attribute
        :param sk_name: Name of the sort key attribute
        :param exclude_keys: Name of the attributes (columns) which should not be compressed. All other attributes
        will be compressed into a single attribute named "v".
        """
        self._pk_name = pk_name
        self._sk_name = sk_name
        
        self._exclude_keys = exclude_keys
    
    def compress(self, ddb_item: Dict):
        exclude_keys = [self._pk_name, self._sk_name]
        exclude_keys.extend(self._exclude_keys)
        attribs_to_compress = {k: v for k, v in ddb_item.items() if k not in exclude_keys}
        uncompressed_attribs = {k: v for k, v in ddb_item.items() if k in exclude_keys}
        
        compressed_attribs = {}
        if attribs_to_compress:
            compressed_attribs = {"v": bz2.compress(pickle.dumps(attribs_to_compress))}
        
        return {
            self._pk_name: ddb_item[self._pk_name],
            self._sk_name: ddb_item[self._sk_name],
            **compressed_attribs,
            **uncompressed_attribs,
        }
    
    def decompress(self, item: Dict):
        
        excluded_attribs = {k: v for k, v in item.items() if k != 'v'}
        
        compressed_value = item.get('v', None)
        
        decompressed_attribs = {}
        if compressed_value:
            v = compressed_value
            if isinstance(v, Binary):
                v = bytes(v)
            decompressed_attribs = pickle.loads(bz2.decompress(v))
        
        return_item = {
            **excluded_attribs,
            **decompressed_attribs,
        }
        return return_item
