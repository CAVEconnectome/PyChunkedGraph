import math


class KeyTranslator:
    """
    Translator class that provides a set of methods to translate between the internal DynamoDB composite primary key
    - partition key (pk) and sort key (sk) into a unified key used by the client code and vice versa.
    """
    
    def __init__(self):
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
    
    def to_unified_key(self, pk, sk):
        return sk.encode()
    
    def to_pk_sk(self, key: bytes):
        prefix, ikey, suffix = self._to_int_key_parts(key)
        sk = key.decode()
        if ikey is not None:
            pk = self._int_key_to_pk(ikey)
        else:
            pk = key.decode()
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
                prefix_start, ikey_start, suffix_start = self._to_int_key_parts(start_key)
                sk_start = self._from_int_key_parts(prefix_start, ikey_start + 1, suffix_start)
        
        if sk_end is not None:
            if not end_inclusive:
                prefix_end, ikey_end, suffix_end = self._to_int_key_parts(end_key)
                sk_end = self._from_int_key_parts(prefix_end, ikey_end - 1, suffix_end)
        
        return pk_start if pk_start is not None else pk_end, sk_start, sk_end
    
    def _to_int_key_parts(self, key: bytes):
        """
        # A utility method to split the given key into prefix, an integer key, and a suffix
        # E.g.,
        # - 00076845692567897775 -> prefix = None, ikey = 76845692567897775, suffix = None
        # - f00144821212986474496 -> prefix = "f", ikey = 144821212986474496, suffix = None
        # - i00145242668664881152 -> prefix = "i", ikey = 145242668664881152, suffix = None
        # - i00216172782113783808_237 -> prefix = "i", ikey = 216172782113783808, suffix = "237"
        # - foperations -> prefix = "f", ikey = None, suffix = None
        # - meta -> prefix = None, ikey = None, suffix = None
        #
        :param key:
        :return:
        """
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
    
    def _from_int_key_parts(self, prefix, ikey, suffix, delim="_"):
        suffix_str = '' if suffix is None else f"{delim}{suffix}"
        return f"{'' if prefix is None else prefix}{ikey}{suffix_str}"
    
    def _int_key_to_pk(self, ikey: int):
        return f"{(ikey >> self._pk_key_shift):{self._pk_int_format}}"
