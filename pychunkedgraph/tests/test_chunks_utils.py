"""Tests for pychunkedgraph.graph.chunks.utils"""

import numpy as np
import pytest

from pychunkedgraph.graph.chunks import utils as chunk_utils


class TestGetChunkLayer:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        from .helpers import to_label

        node_id = to_label(graph, 1, 0, 0, 0, 1)
        assert chunk_utils.get_chunk_layer(graph.meta, node_id) == 1

    def test_higher_layer(self, gen_graph):
        graph = gen_graph(n_layers=4)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=3, x=0, y=0, z=0)
        assert chunk_utils.get_chunk_layer(graph.meta, chunk_id) == 3


class TestGetChunkLayers:
    def test_empty(self, gen_graph):
        graph = gen_graph(n_layers=4)
        result = chunk_utils.get_chunk_layers(graph.meta, [])
        assert len(result) == 0

    def test_multiple(self, gen_graph):
        graph = gen_graph(n_layers=4)
        from .helpers import to_label

        ids = [
            to_label(graph, 1, 0, 0, 0, 1),
            to_label(graph, 1, 1, 0, 0, 2),
        ]
        layers = chunk_utils.get_chunk_layers(graph.meta, ids)
        np.testing.assert_array_equal(layers, [1, 1])


class TestGetChunkCoordinates:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=1, y=2, z=3)
        coords = chunk_utils.get_chunk_coordinates(graph.meta, chunk_id)
        np.testing.assert_array_equal(coords, [1, 2, 3])


class TestGetChunkCoordinatesMultiple:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        ids = [
            chunk_utils.get_chunk_id(graph.meta, layer=2, x=0, y=0, z=0),
            chunk_utils.get_chunk_id(graph.meta, layer=2, x=1, y=2, z=3),
        ]
        coords = chunk_utils.get_chunk_coordinates_multiple(graph.meta, ids)
        np.testing.assert_array_equal(coords[0], [0, 0, 0])
        np.testing.assert_array_equal(coords[1], [1, 2, 3])

    def test_empty(self, gen_graph):
        graph = gen_graph(n_layers=4)
        result = chunk_utils.get_chunk_coordinates_multiple(graph.meta, [])
        assert result.shape == (0, 3)


class TestGetChunkId:
    def test_from_node_id(self, gen_graph):
        graph = gen_graph(n_layers=4)
        from .helpers import to_label

        node_id = to_label(graph, 1, 2, 3, 1, 5)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, node_id=node_id)
        coords = chunk_utils.get_chunk_coordinates(graph.meta, chunk_id)
        np.testing.assert_array_equal(coords, [2, 3, 1])

    def test_from_components(self, gen_graph):
        graph = gen_graph(n_layers=4)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=2, x=1, y=2, z=3)
        assert chunk_utils.get_chunk_layer(graph.meta, chunk_id) == 2
        coords = chunk_utils.get_chunk_coordinates(graph.meta, chunk_id)
        np.testing.assert_array_equal(coords, [1, 2, 3])


class TestComputeChunkIdOutOfRange:
    def test_raises(self, gen_graph):
        graph = gen_graph(n_layers=4)
        with pytest.raises(ValueError, match="out of range"):
            chunk_utils._compute_chunk_id(graph.meta, layer=2, x=9999, y=0, z=0)


class TestGetChunkIdsFromCoords:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        result = chunk_utils.get_chunk_ids_from_coords(graph.meta, 2, coords)
        assert len(result) == 2
        for cid in result:
            assert chunk_utils.get_chunk_layer(graph.meta, cid) == 2


class TestGetChunkIdsFromNodeIds:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        from .helpers import to_label

        ids = np.array(
            [
                to_label(graph, 1, 0, 0, 0, 1),
                to_label(graph, 1, 1, 0, 0, 2),
            ],
            dtype=np.uint64,
        )
        result = chunk_utils.get_chunk_ids_from_node_ids(graph.meta, ids)
        assert len(result) == 2

    def test_empty(self, gen_graph):
        graph = gen_graph(n_layers=4)
        result = chunk_utils.get_chunk_ids_from_node_ids(graph.meta, [])
        assert len(result) == 0


class TestNormalizeBoundingBox:
    def test_none(self, gen_graph):
        graph = gen_graph(n_layers=4)
        assert chunk_utils.normalize_bounding_box(graph.meta, None, False) is None


class TestGetBoundingChildrenChunks:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=5)
        result = chunk_utils.get_bounding_children_chunks(graph.meta, 3, (0, 0, 0), 2)
        assert len(result) > 0
        assert result.shape[1] == 3
