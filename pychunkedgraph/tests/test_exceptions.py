"""Tests for pychunkedgraph.graph.exceptions"""

import pytest
from http.client import BAD_REQUEST, UNAUTHORIZED, FORBIDDEN, CONFLICT
from http.client import INTERNAL_SERVER_ERROR, GATEWAY_TIMEOUT

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
    SupervoxelSplitRequiredError,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        with pytest.raises(ChunkedGraphError):
            raise ChunkedGraphError("test")

    def test_locking_error_inherits(self):
        assert issubclass(LockingError, ChunkedGraphError)
        with pytest.raises(ChunkedGraphError):
            raise LockingError("locked")

    def test_precondition_error(self):
        assert issubclass(PreconditionError, ChunkedGraphError)

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


class TestSupervoxelSplitRequiredError:
    def test_inherits_chunkedgraph_error(self):
        assert issubclass(SupervoxelSplitRequiredError, ChunkedGraphError)

    def test_stores_sv_remapping(self):
        remap = {1: 10, 2: 20}
        err = SupervoxelSplitRequiredError("split needed", remap)
        assert err.sv_remapping == remap
        assert str(err) == "split needed"

    def test_stores_operation_id(self):
        err = SupervoxelSplitRequiredError("msg", {}, operation_id=42)
        assert err.operation_id == 42

    def test_operation_id_default_none(self):
        err = SupervoxelSplitRequiredError("msg", {})
        assert err.operation_id is None

    def test_can_be_caught_as_chunkedgraph_error(self):
        with pytest.raises(ChunkedGraphError):
            raise SupervoxelSplitRequiredError("test", {1: 2})
