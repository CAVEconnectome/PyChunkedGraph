"""Deterministic, scattered chunk ordering for the per-layer Indexed Job.

WHY SCATTER. A layer's chunks form an ``X*Y*Z`` grid. In row-major order the
flat index ``x*(Y*Z) + y*Z + z`` makes consecutive indices spatially-adjacent
chunks, whose Bigtable row keys (derived from chunk coords) fall in a narrow key
range. Processing chunks in flat order would point the whole worker fleet at one
tablet at a time -> a write hotspot. So we never process in flat order.

HOW. k8s hands batch ``i`` the contiguous position window ``[i*B, (i+1)*B)``.
``permute`` maps each position to a pseudo-random chunk flat-index (a bijection
over ``[0, N)``), so the chunks one batch touches — and the chunks of all batches
running at once — spread uniformly across the whole grid / key range.

EXAMPLE (N = 100_000, a 50x50x40 grid, seed = 42). The first eight sequence
positions map to flat indices spanning 17_224..92_229 — nearly the entire
``[0, 100_000)`` range — e.g.:
    pos 0 -> flat 19_165 -> chunk (9, 29, 5)
    pos 1 -> flat 74_379 -> chunk (37, 9, 19)
    pos 2 -> flat 92_229 -> chunk (46, 5, 29)
    pos 3 -> flat 89_707 -> chunk (44, 42, 27)
    pos 4 -> flat 30_659 -> chunk (15, 16, 19)
Adjacent positions land far apart, so a single pod's batch — and the fleet as a
whole — hits scattered tablets rather than one hot range.

PROPERTIES. ``permute`` is a balanced Feistel network with cycle-walking: a true
bijection over ``[0, N)`` needing no materialized array (scales to billions of
chunks); a pure function of ``(pos, N, seed)`` so a retry of position i hits the
exact same chunk; and invertible (``unpermute``) to turn a failed index back into
its coord for inspection. The seed is pinned per ingest run.
"""

from typing import List, Sequence, Tuple

import numpy as np

_ROUNDS = 4
_U64 = 0xFFFFFFFFFFFFFFFF


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return z ^ (z >> 31)


def _round_keys(seed: int) -> Tuple[int, ...]:
    return tuple(_splitmix64(seed + i) for i in range(_ROUNDS))


def _half_bits(n: int) -> int:
    """Half the (even) bit width of the smallest power-of-two domain >= n."""
    total = max(2, (n - 1).bit_length())
    return (total + 1) // 2


def _mix(value: int, round_key: int, mask: int) -> int:
    return _splitmix64(value ^ round_key) & mask


def _feistel_forward(x: int, half_bits: int, round_keys: Tuple[int, ...]) -> int:
    mask = (1 << half_bits) - 1
    left, right = (x >> half_bits) & mask, x & mask
    for rk in round_keys:
        left, right = right, left ^ _mix(right, rk, mask)
    return (left << half_bits) | right


def _feistel_inverse(y: int, half_bits: int, round_keys: Tuple[int, ...]) -> int:
    mask = (1 << half_bits) - 1
    left, right = (y >> half_bits) & mask, y & mask
    for rk in reversed(round_keys):
        left, right = right ^ _mix(left, rk, mask), left
    return (left << half_bits) | right


def permute(pos: int, n: int, seed: int) -> int:
    """Map sequence position ``pos`` -> a scattered chunk flat-index in ``[0, n)``."""
    if n <= 1:
        return 0
    half_bits, keys = _half_bits(n), _round_keys(seed)
    x = pos
    while True:
        x = _feistel_forward(x, half_bits, keys)
        if x < n:
            return x


def unpermute(flat: int, n: int, seed: int) -> int:
    """Inverse of ``permute``: chunk flat-index -> its sequence position."""
    if n <= 1:
        return 0
    half_bits, keys = _half_bits(n), _round_keys(seed)
    x = flat
    while True:
        x = _feistel_inverse(x, half_bits, keys)
        if x < n:
            return x


def num_batches(n: int, batch_size: int) -> int:
    return (n + batch_size - 1) // batch_size


def batch_coords(
    batch_index: int, bounds: Sequence[int], seed: int, batch_size: int
) -> List[Tuple[int, int, int]]:
    """The chunk coords for one batch index (its slice of the shuffled order)."""
    shape = (int(bounds[0]), int(bounds[1]), int(bounds[2]))
    n = shape[0] * shape[1] * shape[2]
    start = batch_index * batch_size
    out = []
    for pos in range(start, min(start + batch_size, n)):
        x, y, z = np.unravel_index(permute(pos, n, seed), shape)
        out.append((int(x), int(y), int(z)))
    return out


def coord_to_batch_index(
    coord: Sequence[int], bounds: Sequence[int], seed: int, batch_size: int
) -> int:
    """Which batch a chunk coord belongs to (for failed-index inspection)."""
    shape = (int(bounds[0]), int(bounds[1]), int(bounds[2]))
    n = shape[0] * shape[1] * shape[2]
    flat = int(np.ravel_multi_index((int(coord[0]), int(coord[1]), int(coord[2])), shape))
    return unpermute(flat, n, seed) // batch_size
