"""A fork that cannot be made is refused before any write, and the admin endpoint forks a
table at the requested time."""

import inspect
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from flask import Flask
from kvdbclient.fork.marker import Hop

from ...app.segmentation import common
from ...app.segmentation.v1 import routes
from ...graph.exceptions import BadRequest, PreconditionError
from ...graph.meta import ChunkedGraphMeta

#: The fork time the stand-in fork's marker holds.
TF = datetime(2026, 11, 25, tzinfo=UTC)


def _opened() -> SimpleNamespace:
    """A stand-in for the fork `create_fork` opens, its client recording its close."""
    client = SimpleNamespace(marker=SimpleNamespace(chain=(Hop("source", TF),)))
    client.closed = False
    client.close = lambda: setattr(client, "closed", True)
    return SimpleNamespace(graph_id="fork", client=client)


def _handle(body, query=""):
    """The fork handler's answer to one POST of `body`, with `query` as its args."""
    with Flask(__name__).test_request_context(f"/?{query}", method="POST", json=body):
        return common.handle_fork("source")


def _recorded(monkeypatch) -> list[tuple[str, str, datetime]]:
    """The calls the handler makes to `create_fork`, stubbed at that boundary."""
    calls = []

    def created(*args, **_):
        calls.append(args)
        return _opened()

    monkeypatch.setattr(common, "create_fork", created)
    return calls


def test_a_second_fork_under_one_name_is_refused(joined, fork_of):
    cg, _ = joined
    fork_of(cg, "test_fork", datetime.now(UTC))
    with pytest.raises(PreconditionError, match="test_fork already exists"):
        fork_of(cg, "test_fork", datetime.now(UTC))


def test_a_source_splitting_supervoxels_in_ocdbt_is_refused_writing_nothing(
    joined, fork_of, monkeypatch
):
    cg, _ = joined
    monkeypatch.setattr(ChunkedGraphMeta, "ocdbt_seg", property(lambda _: True))
    with pytest.raises(PreconditionError, match="splits supervoxels in OCDBT"):
        fork_of(cg, "test_fork", datetime.now(UTC))
    monkeypatch.setattr(ChunkedGraphMeta, "ocdbt_seg", property(lambda _: False))
    assert fork_of(cg, "test_fork", datetime.now(UTC)).graph_id == "test_fork"


def test_the_endpoint_forks_at_the_requested_time(monkeypatch):
    calls = _recorded(monkeypatch)
    _handle({"fork_id": "fork"}, "timestamp=1700000000")
    assert calls == [("source", "fork", datetime.fromtimestamp(1700000000, UTC))]


def test_the_endpoint_forks_now_without_a_timestamp(monkeypatch):
    calls = _recorded(monkeypatch)
    before = datetime.now(UTC)
    _handle({"fork_id": "fork"})
    assert before <= calls[0][2] <= datetime.now(UTC)


def test_the_endpoint_answers_the_fork_time_and_closes_the_fork(monkeypatch):
    fork = _opened()
    monkeypatch.setattr(common, "create_fork", lambda *args, **_: fork)
    assert _handle({"fork_id": "fork"}) == common.CreatedFork("fork", TF)
    assert fork.client.closed


def test_the_endpoint_needs_a_fork_id(monkeypatch):
    calls = _recorded(monkeypatch)
    with pytest.raises(BadRequest, match="fork_id"):
        _handle({})
    assert calls == []


def test_a_refused_fork_answers_400_with_its_reason(monkeypatch):
    def refuse(*args, **kwargs):
        raise PreconditionError(
            "source splits supervoxels in OCDBT, which forks cannot"
        )

    monkeypatch.setattr(common, "create_fork", refuse)
    with pytest.raises(BadRequest, match="splits supervoxels in OCDBT") as refused:
        _handle({"fork_id": "fork"})
    assert refused.value.status_code == 400


def test_the_route_answers_the_fork_its_source_and_its_time(monkeypatch):
    created = common.CreatedFork("fork", TF)
    monkeypatch.setattr(common, "handle_fork", lambda table_id: created)
    with Flask(__name__).test_request_context(method="POST"):
        answer = inspect.unwrap(routes.handle_fork)("source")
    expected = {"fork_id": "fork", "source": "source", "fork_time": TF.timestamp()}
    assert answer.get_json() == expected
