"""Self-contained per-chunk lock on Bigtable: at most one effective writer per chunk.

A dedicated, hash-prefixed lock row per chunk_id holds two cells in the
Concurrency family: a ``claim`` (token + cell-timestamp, expiry-based) and a
``done`` marker. Acquire/mark-done/renew/release are atomic ``conditional_row``
CAS ops mirroring ``BigTableClient.lock_root``. All writes are value-matched on
the claim token (a fence) so a zombie whose claim was stolen cannot mark done.
Family "0" keeps all cell versions, so token/freshness checks look only at the
latest claim via ``CellsColumnLimitFilter(1)``.
"""

import hashlib
from datetime import datetime, timedelta, timezone

from google.cloud.bigtable.row_filters import (
    CellsColumnLimitFilter,
    ColumnRangeFilter,
    RowFilterChain,
    RowFilterUnion,
    TimestampRange,
    TimestampRangeFilter,
    ValueRangeFilter,
)

ACQUIRED = "acquired"
DONE = "done"
HELD = "held"

_FAMILY = "0"
_CLAIM = b"claim"
_DONE = b"done"


def _row_key(chunk_id: int) -> bytes:
    packed = int(chunk_id).to_bytes(8, "big")
    return hashlib.blake2b(packed, digest_size=2).digest() + packed


def _b(token: int) -> bytes:
    return int(token).to_bytes(8, "big")


def _now() -> datetime:
    """Bigtable-compatible 'now': UTC, rounded down to the millisecond."""
    t = datetime.now(timezone.utc)
    return t - timedelta(microseconds=t.microsecond % 1000)


def _col(column: bytes) -> ColumnRangeFilter:
    return ColumnRangeFilter(
        _FAMILY, start_column=column, end_column=column,
        inclusive_start=True, inclusive_end=True,
    )


def _claim_is(token: int) -> RowFilterChain:
    """Latest claim cell exists and its value == token (the fence)."""
    return RowFilterChain([_col(_CLAIM), CellsColumnLimitFilter(1),
                           ValueRangeFilter(start_value=_b(token), end_value=_b(token),
                                            inclusive_start=True, inclusive_end=True)])


def _is_done(table, row_key: bytes) -> bool:
    return table.read_row(row_key, filter_=_col(_DONE)) is not None


def acquire(table, chunk_id: int, token: int, expiry: timedelta) -> str:
    """Try to claim a chunk. Returns ACQUIRED, DONE (skip), or HELD (retry)."""
    row_key = _row_key(chunk_id)
    cutoff = _now() - expiry
    fresh_claim = RowFilterChain(
        [_col(_CLAIM), CellsColumnLimitFilter(1),
         TimestampRangeFilter(TimestampRange(start=cutoff))]
    )
    base = RowFilterUnion([_col(_DONE), fresh_claim])
    row = table.conditional_row(row_key, filter_=base)
    # set_cell applies when the base condition is FALSE (no done, no live claim)
    row.set_cell(_FAMILY, _CLAIM, _b(token), state=False,
                 timestamp=_now())
    blocked = row.commit()
    if not blocked:
        return ACQUIRED
    return DONE if _is_done(table, row_key) else HELD


def mark_done(table, chunk_id: int, token: int) -> bool:
    """Mark a chunk done iff we still hold the claim. Returns False if fenced out."""
    row = table.conditional_row(_row_key(chunk_id), filter_=_claim_is(token))
    row.set_cell(_FAMILY, _DONE, _b(token), state=True,
                 timestamp=_now())
    row.delete_cell(_FAMILY, _CLAIM, state=True)
    return bool(row.commit())


def renew(table, chunk_id: int, token: int) -> bool:
    """Extend our claim's expiry. Returns False if we no longer hold it."""
    row = table.conditional_row(_row_key(chunk_id), filter_=_claim_is(token))
    row.set_cell(_FAMILY, _CLAIM, _b(token), state=True,
                 timestamp=_now())
    return bool(row.commit())


def release(table, chunk_id: int, token: int) -> None:
    """Drop our claim (best-effort) so a sequential retry can re-claim before expiry."""
    row = table.conditional_row(_row_key(chunk_id), filter_=_claim_is(token))
    row.delete_cell(_FAMILY, _CLAIM, state=True)
    row.commit()
