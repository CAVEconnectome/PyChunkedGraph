from six.moves import http_client


class ChunkedGraphError(Exception):
    """Base class for all exceptions raised by the ChunkedGraph"""

    pass


class LockingError(ChunkedGraphError):
    """Raised when a Bigtable Lock could not be acquired"""

    pass


class PreconditionError(ChunkedGraphError):
    """Raised when preconditions for Chunked Graph operations are not met"""

    pass


class PostconditionError(ChunkedGraphError):
    """Raised when postconditions for Chunked Graph operations are not met"""

    pass


class ChunkedGraphAPIError(ChunkedGraphError):
    """Base class for exceptions raised by calling API methods.

    Args:
        message (str): The exception message.
    """

    status_code = None
    """Optional[int]: The HTTP status code associated with this error.

    This may be ``None`` if the exception does not have a direct mapping
    to an HTTP error.

    See http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
    """

    def __init__(self, message):
        super(ChunkedGraphAPIError, self).__init__(message)
        self.message = message

    def __str__(self):
        return f"[{self.status_code}]: {self.message}"


class ClientError(ChunkedGraphAPIError):
    """Base class for all client error (HTTP 4xx) responses."""


class BadRequest(ClientError):
    """Exception mapping a ``400 Bad Request`` response."""

    status_code = http_client.BAD_REQUEST


class Unauthorized(ClientError):
    """Exception mapping a ``401 Unauthorized`` response."""

    status_code = http_client.UNAUTHORIZED


class Forbidden(ClientError):
    """Exception mapping a ``403 Forbidden`` response."""

    status_code = http_client.FORBIDDEN


class Conflict(ClientError):
    """Exception mapping a ``409 Conflict`` response."""

    status_code = http_client.CONFLICT


class ServerError(ChunkedGraphAPIError):
    """Base for 5xx responses."""


class InternalServerError(ServerError):
    """Exception mapping a ``500 Internal Server Error`` response."""

    status_code = http_client.INTERNAL_SERVER_ERROR


class GatewayTimeout(ServerError):
    """Exception mapping a ``504 Gateway Timeout`` response."""

    status_code = http_client.GATEWAY_TIMEOUT


class SupervoxelSplitRequiredError(ChunkedGraphError):
    """
    Raised when supervoxel splitting is necessary.
    Edit process should catch this error and retry after supervoxel has been split.
    Saves remapping required for detecting which supervoxels need to be split.
    """

    def __init__(
        self,
        message: str,
        sv_remapping: dict,
        operation_id: int | None = None,
    ):
        super().__init__(message)
        self.sv_remapping = sv_remapping
        self.operation_id = operation_id
