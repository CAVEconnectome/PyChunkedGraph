"""
Common client util functions
"""


def get_default_client_info():
    """
    Load client from env variables.
    """

    # TODO make dynamic after multiple platform support is added
    from .bigtable.utils import get_bigtable_client_info
    from ..meta import BackendClientInfo

    return BackendClientInfo(
        CONFIG=get_bigtable_client_info(admin=True, read_only=False)
    )
