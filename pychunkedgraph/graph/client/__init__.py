"""
Sub packages/modules for backend storage clients
Currently supports Google Big Table and HBase

A simple client needs to be able to create the graph,
store graph meta and to write and read node information.
Also needs locking support to prevent race conditions
when modifying root/parent nodes.

In addition, clients with more features like generating unique IDs
and logging facilities can be implemented by inherting respective base classes.

These methods are in separate classes because they are logically related.
This also makes it possible to have different backend storage solutions,
making it possible to use any unique features these solutions may provide.

Please see `base.py` for more details.
"""

from collections import namedtuple

from .bigtable.client import Client as BigTableClient
from .hbase.client import Client as HBaseClient


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = (None, None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)


def get_client_from_info(table_id: str, client_info: BackendClientInfo, graph_meta=None):
    """
    Factory function to create appropriate client based on BackendClientInfo.
    
    Args:
        table_id: Table/graph identifier
        client_info: BackendClientInfo with TYPE and CONFIG
        graph_meta: Optional ChunkedGraphMeta
        
    Returns:
        Client instance (BigTableClient or HBaseClient)
    """
    backend_type = client_info.TYPE or "bigtable"  # Default to bigtable
    
    if backend_type.lower() == "bigtable":
        return BigTableClient(table_id, config=client_info.CONFIG, graph_meta=graph_meta)
    elif backend_type.lower() == "hbase":
        return HBaseClient(table_id, config=client_info.CONFIG, graph_meta=graph_meta)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def get_default_client_info():
    """
    Load client from env variables.
    """
    import os
    backend_type = os.environ.get("BACKEND_TYPE", "bigtable").lower()
    
    if backend_type == "hbase":
        from .hbase import get_client_info as get_hbase_client_info
        return BackendClientInfo(
            TYPE="hbase",
            CONFIG=get_hbase_client_info(admin=True, read_only=False)
        )
    else:
        from .bigtable import get_client_info as get_bigtable_client_info
        return BackendClientInfo(
            TYPE="bigtable",
            CONFIG=get_bigtable_client_info(admin=True, read_only=False)
        )
