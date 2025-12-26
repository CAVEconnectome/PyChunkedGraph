from collections import namedtuple
from os import environ

DEFAULT_TABLE_PREFIX = "neuromancer-seung-import.pychunkedgraph"
DEFAULT_AWS_REGION = "us-east-1"

_amazon_dynamodb_config_fields = (
    "REGION",
    "TABLE_PREFIX",
    "ADMIN",
    "READ_ONLY",
    "END_POINT",
)
_amazon_dynamodb_config_defaults = (
    environ.get("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION),
    environ.get("AMAZON_DYNAMODB_TABLE_PREFIX", DEFAULT_TABLE_PREFIX),
    False,
    True,
    None,
)
AmazonDynamoDbConfig = namedtuple(
    "AmazonDynamoDbConfig", _amazon_dynamodb_config_fields, defaults=_amazon_dynamodb_config_defaults
)


def get_client_info(
    region: str = None,
    table_prefix: str = None,
    admin: bool = False,
    read_only: bool = True,
):
    """Helper function to load config from env."""
    _region = region if region else environ.get("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION)
    _table_prefix = table_prefix if table_prefix else environ.get("AMAZON_DYNAMODB_TABLE_PREFIX", DEFAULT_TABLE_PREFIX)

    kwargs = {
        "REGION": _region,
        "TABLE_PREFIX": _table_prefix,
        "ADMIN": admin,
        "READ_ONLY": read_only
    }
    return AmazonDynamoDbConfig(**kwargs)
