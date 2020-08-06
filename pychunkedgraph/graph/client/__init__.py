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

from collections import namedtuple


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = (None, None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)


def get_default_client_info():
    """
    Load client from env variables.
    """

    # TODO make dynamic after multiple platform support is added
    from .bigtable import get_client_info as get_bigtable_client_info

    return BackendClientInfo(
        CONFIG=get_bigtable_client_info(admin=True, read_only=False)
    )
