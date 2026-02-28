"""Tests for pychunkedgraph.graph.exceptions"""

import pytest
from http.client import BAD_REQUEST, UNAUTHORIZED, FORBIDDEN, CONFLICT
from http.client import INTERNAL_SERVER_ERROR, GATEWAY_TIMEOUT

from kvdbclient.exceptions import KVDBClientError
from pychunkedgraph.graph.exceptions import (
    ChunkedGraphError,
    LockingError,
    PreconditionError,
    PostconditionError,
    ChunkedGraphAPIError,
    ClientError,
    BadRequest,
    Unauthorized,
    Forbidden,
    Conflict,
    ServerError,
    InternalServerError,
    GatewayTimeout,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        with pytest.raises(ChunkedGraphError):
            raise ChunkedGraphError("test")

    def test_locking_error_inherits(self):
        assert issubclass(LockingError, KVDBClientError)
        with pytest.raises(KVDBClientError):
            raise LockingError("locked")

    def test_precondition_error(self):
        assert issubclass(PreconditionError, KVDBClientError)

    def test_postcondition_error(self):
        assert issubclass(PostconditionError, ChunkedGraphError)

    def test_api_error_str(self):
        err = ChunkedGraphAPIError("test message")
        assert err.message == "test message"
        assert err.status_code is None
        assert "[None]: test message" == str(err)

    def test_client_error_inherits(self):
        assert issubclass(ClientError, ChunkedGraphAPIError)

    def test_bad_request(self):
        err = BadRequest("bad")
        assert err.status_code == BAD_REQUEST
        assert issubclass(BadRequest, ClientError)

    def test_unauthorized(self):
        assert Unauthorized.status_code == UNAUTHORIZED

    def test_forbidden(self):
        assert Forbidden.status_code == FORBIDDEN

    def test_conflict(self):
        assert Conflict.status_code == CONFLICT

    def test_server_error_inherits(self):
        assert issubclass(ServerError, ChunkedGraphAPIError)

    def test_internal_server_error(self):
        assert InternalServerError.status_code == INTERNAL_SERVER_ERROR

    def test_gateway_timeout(self):
        assert GatewayTimeout.status_code == GATEWAY_TIMEOUT
