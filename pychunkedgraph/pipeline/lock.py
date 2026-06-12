"""Self-contained per-chunk lock, built on kvdbclient primitives.

At most one effective writer per chunk. A dedicated, hash-prefixed lock row per
chunk_id holds two cells in family "0": kvdbclient's expiry-fenced ``Lock`` cell
(the claim, via ``lock_by_row_key``/``renew_lock_by_row_key``/``unlock_by_row_key``,
all value-matched on the token so a zombie can't act under a stolen claim) and a
local ``done`` cell written/read through ``mutate_row``/``write``/``_read_byte_row``
so a re-run skips finished chunks. Claim expiry is the kvdbclient client's
``_lock_expiry`` (set at construction), not a per-call value.
"""

import hashlib

import numpy as np
from kvdbclient import attributes, serializers

ACQUIRED = "acquired"
DONE = "done"
HELD = "held"

# Local "done" marker on the lock row; family "0" matches kvdbclient's Lock cell.
# Constructing the _Attribute self-registers it so _read_byte_row can deserialize.
_DONE = attributes._Attribute(
    key=b"chunk_done", family_id="0", serializer=serializers.UInt64String()
)


def _row_key(chunk_id: int) -> bytes:
    packed = int(chunk_id).to_bytes(8, "big")
    return hashlib.blake2b(packed, digest_size=2).digest() + packed


def _op(token: int) -> np.uint64:
    return np.uint64(int(token))


def _is_done(client, row_key: bytes) -> bool:
    return bool(client._read_byte_row(row_key, columns=_DONE))


def acquire(client, chunk_id: int, token: int, expiry=None) -> str:
    """Try to claim a chunk. Returns ACQUIRED, DONE (skip), or HELD (retry).
    `expiry` is ignored: kvdbclient uses the client's configured lock expiry."""
    row_key = _row_key(chunk_id)
    if _is_done(client, row_key):
        return DONE
    if client.lock_by_row_key(row_key, _op(token)):
        return ACQUIRED
    return DONE if _is_done(client, row_key) else HELD


def mark_done(client, chunk_id: int, token: int) -> bool:
    """Mark a chunk done iff we still hold the claim. Returns False if fenced out."""
    row_key = _row_key(chunk_id)
    if not client.renew_lock_by_row_key(row_key, _op(token)):
        return False  # claim lost -> someone else owns the chunk; don't mark
    client.write([client.mutate_row(row_key, {_DONE: int(token)})])
    client.unlock_by_row_key(row_key, _op(token))
    return True


def renew(client, chunk_id: int, token: int) -> bool:
    """Extend our claim's expiry. Returns False if we no longer hold it."""
    return bool(client.renew_lock_by_row_key(_row_key(chunk_id), _op(token)))


def release(client, chunk_id: int, token: int) -> None:
    """Drop our claim (best-effort) so a sequential retry can re-claim before expiry."""
    client.unlock_by_row_key(_row_key(chunk_id), _op(token))
