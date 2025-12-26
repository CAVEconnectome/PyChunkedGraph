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

from .base import SimpleClient


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = (None, None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)
GCP_BIGTABLE_BACKEND_TYPE = "bigtable"
AMAZON_DYNAMODB_BACKEND_TYPE = "amazon.dynamodb"
DEFAULT_BACKEND_TYPE = GCP_BIGTABLE_BACKEND_TYPE
SUPPORTED_BACKEND_TYPES={GCP_BIGTABLE_BACKEND_TYPE, AMAZON_DYNAMODB_BACKEND_TYPE}

def get_default_client_info():
    """
    Get backend client type from BACKEND_CLIENT_TYPE env variable.
    """
    backend_type_env = environ.get("BACKEND_CLIENT_TYPE", DEFAULT_BACKEND_TYPE)
    if backend_type_env == GCP_BIGTABLE_BACKEND_TYPE:
        from .bigtable import get_client_info as get_bigtable_client_info
        client_info = BackendClientInfo(
            TYPE=backend_type_env,
            CONFIG=get_bigtable_client_info(admin=True, read_only=False)
        )
    elif backend_type_env == AMAZON_DYNAMODB_BACKEND_TYPE:
        from .amazon.dynamodb import get_client_info as get_amazon_dynamodb_client_info
        client_info = BackendClientInfo(
            TYPE=backend_type_env,
            CONFIG=get_amazon_dynamodb_client_info(admin=True, read_only=False)
        )
    else:
        raise TypeError(f"Client backend {backend_type_env} is not supported, supported backend types: {', '.join(list(SUPPORTED_BACKEND_TYPES))}")
    return client_info

def get_client_class(client_info: BackendClientInfo):
    if isinstance(client_info.TYPE, SimpleClient):
        return client_info.TYPE

    if client_info.TYPE is None:
        class_type = DEFAULT_BACKEND_TYPE
    elif isinstance(client_info.TYPE, str):
        class_type = client_info.TYPE
    else:
        raise TypeError(f"Unsupported client backend {type(client_info.TYPE)}")

    if class_type == GCP_BIGTABLE_BACKEND_TYPE:
        from .bigtable.client import Client as BigTableClient
        ret_class_type = BigTableClient
    elif class_type == AMAZON_DYNAMODB_BACKEND_TYPE:
        from .amazon.dynamodb.client import Client as AmazonDynamoDbClient
        ret_class_type = AmazonDynamoDbClient
    else:
        raise TypeError(f"Client backend {class_type} is not supported, supported backend types: {', '.join(list(SUPPORTED_BACKEND_TYPES))}")

    return ret_class_type
