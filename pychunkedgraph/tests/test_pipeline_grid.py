"""Pure unit tests for the Indexed-Job chunk permutation (no external deps)."""

import numpy as np

from ..pipeline import grid


def _all_coords(bounds, seed, batch_size):
    n = bounds[0] * bounds[1] * bounds[2]
    coords = []
    for i in range(grid.num_batches(n, batch_size)):
        coords.extend(grid.batch_coords(i, bounds, seed, batch_size))
    return coords


def test_permutation_is_bijection():
    for n in [1, 2, 7, 64, 100, 513, 1000]:
        assert sorted(grid.permute(p, n, seed=42) for p in range(n)) == list(range(n))


def test_permute_unpermute_inverse():
    n, seed = 1000, 7
    assert all(grid.unpermute(grid.permute(p, n, seed), n, seed) == p for p in range(n))


def test_deterministic_per_seed():
    n = 257
    assert [grid.permute(p, n, 123) for p in range(n)] == [
        grid.permute(p, n, 123) for p in range(n)
    ]
    assert [grid.permute(p, n, 123) for p in range(n)] != [
        grid.permute(p, n, 124) for p in range(n)
    ]


def test_batches_partition_grid_exactly():
    bounds, seed, batch_size = (4, 5, 3), 99, 7  # n = 60
    coords = _all_coords(bounds, seed, batch_size)
    assert len(coords) == 60
    assert len(set(coords)) == 60
    assert all(0 <= x < 4 and 0 <= y < 5 and 0 <= z < 3 for x, y, z in coords)


def test_coord_to_batch_index_roundtrip():
    bounds, seed, batch_size = (4, 5, 3), 99, 7
    for i in range(grid.num_batches(60, batch_size)):
        for coord in grid.batch_coords(i, bounds, seed, batch_size):
            assert grid.coord_to_batch_index(coord, bounds, seed, batch_size) == i


def test_batch_is_scattered_not_contiguous():
    bounds, seed, batch_size = (16, 16, 16), 5, 16
    coords = grid.batch_coords(0, bounds, seed, batch_size)
    flats = sorted(int(np.ravel_multi_index(c, bounds)) for c in coords)
    # a contiguous raster block would span batch_size-1; scattered spans far wider
    assert flats[-1] - flats[0] > 4 * batch_size
