from collections import namedtuple
from os import environ

DEFAULT_PROJECT = "neuromancer-seung-import"
DEFAULT_INSTANCE = "pychunkedgraph"

_bigtableconfig_fields = (
    "PROJECT",
    "INSTANCE",
    "ADMIN",
    "READ_ONLY",
    "CREDENTIALS",
    "MAX_ROW_KEY_COUNT"
)
_bigtableconfig_defaults = (
    environ.get("BIGTABLE_PROJECT", DEFAULT_PROJECT),
    environ.get("BIGTABLE_INSTANCE", DEFAULT_INSTANCE),
    False,
    True,
    None,
    1000
)
BigTableConfig = namedtuple(
    "BigTableConfig", _bigtableconfig_fields, defaults=_bigtableconfig_defaults
)


def get_client_info(
    project: str = None,
    instance: str = None,
    admin: bool = False,
    read_only: bool = True,
):
    """Helper function to load config from env."""
    _project = environ.get("BIGTABLE_PROJECT", DEFAULT_PROJECT)
    if project:
        _project = project

    _instance = environ.get("BIGTABLE_INSTANCE", DEFAULT_INSTANCE)
    if instance:
        _instance = instance

    kwargs = {
        "PROJECT": _project,
        "INSTANCE": _instance,
        "ADMIN": admin,
        "READ_ONLY": read_only,
    }
    return BigTableConfig(**kwargs)
