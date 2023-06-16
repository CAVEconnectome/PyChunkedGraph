"""
Sub packages/modules for backend storage clients
Currently supports Google Big Table

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

from os import environ
from collections import namedtuple



_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = (None, None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)
DEFAULT_BACKEND_TYPE = "GCP.BIGTABLE"

def get_default_client_info():
    """
    Load client from env variables.
    """
    backend_type_env = environ.get("BACKEND_CLIENT_TYPE", DEFAULT_BACKEND_TYPE)
    if backend_type_env == "GCP.BIGTABLE":
        from .bigtable.client import Client as BigTableClient
        from .bigtable import get_client_info as get_bigtable_client_info
        client_info = BackendClientInfo(
            TYPE=BigTableClient,
            CONFIG=get_bigtable_client_info(admin=True, read_only=False)
        )
    elif backend_type_env == "AMAZON.DYNAMODB":
        from .amazon.dynamodb.client import Client as AmazonDynamoDbClient
        from .amazon.dynamodb import get_client_info as get_amazon_dynamodb_client_info
        client_info = BackendClientInfo(
            TYPE=AmazonDynamoDbClient,
            CONFIG=get_amazon_dynamodb_client_info(admin=True, read_only=False)
        )
    else:
        raise TypeError(f"Client backend {str(backend_type_env)} is not supported")
    return client_info
