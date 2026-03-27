import numpy as np
import pytest

from ..helpers import to_label
from ...graph import serializers


class TestGraphNodeConversion:
    @pytest.mark.timeout(30)
    def test_compute_bitmasks(self, gen_graph):
        cg = gen_graph(n_layers=10)
        # Verify bitmasks for layer and spatial bits
        node_id = cg.get_node_id(np.uint64(1), layer=2, x=0, y=0, z=0)
        assert cg.get_chunk_layer(node_id) == 2
        assert cg.get_segment_id(node_id) == 1

        # Different layers should produce different bitmask regions
        for layer in range(2, 10):
            nid = cg.get_node_id(np.uint64(1), layer=layer, x=0, y=0, z=0)
            assert cg.get_chunk_layer(nid) == layer

    @pytest.mark.timeout(30)
    def test_node_conversion(self, gen_graph):
        cg = gen_graph(n_layers=10)

        node_id = cg.get_node_id(np.uint64(4), layer=2, x=3, y=1, z=0)
        assert cg.get_chunk_layer(node_id) == 2
        assert np.all(cg.get_chunk_coordinates(node_id) == np.array([3, 1, 0]))

        chunk_id = cg.get_chunk_id(layer=2, x=3, y=1, z=0)
        assert cg.get_chunk_layer(chunk_id) == 2
        assert np.all(cg.get_chunk_coordinates(chunk_id) == np.array([3, 1, 0]))

        assert cg.get_chunk_id(node_id=node_id) == chunk_id
        assert cg.get_node_id(np.uint64(4), chunk_id=chunk_id) == node_id

    @pytest.mark.timeout(30)
    def test_node_id_adjacency(self, gen_graph):
        cg = gen_graph(n_layers=10)

        assert cg.get_node_id(np.uint64(0), layer=2, x=3, y=1, z=0) + np.uint64(
            1
        ) == cg.get_node_id(np.uint64(1), layer=2, x=3, y=1, z=0)

        assert cg.get_node_id(
            np.uint64(2**53 - 2), layer=10, x=0, y=0, z=0
        ) + np.uint64(1) == cg.get_node_id(
            np.uint64(2**53 - 1), layer=10, x=0, y=0, z=0
        )

    @pytest.mark.timeout(30)
    def test_serialize_node_id(self, gen_graph):
        cg = gen_graph(n_layers=10)

        assert serializers.serialize_uint64(
            cg.get_node_id(np.uint64(0), layer=2, x=3, y=1, z=0)
        ) < serializers.serialize_uint64(
            cg.get_node_id(np.uint64(1), layer=2, x=3, y=1, z=0)
        )

        assert serializers.serialize_uint64(
            cg.get_node_id(np.uint64(2**53 - 2), layer=10, x=0, y=0, z=0)
        ) < serializers.serialize_uint64(
            cg.get_node_id(np.uint64(2**53 - 1), layer=10, x=0, y=0, z=0)
        )

    @pytest.mark.timeout(30)
    def test_deserialize_node_id(self, gen_graph):
        cg = gen_graph(n_layers=10)
        node_id = cg.get_node_id(np.uint64(4), layer=2, x=3, y=1, z=0)
        serialized = serializers.serialize_uint64(node_id)
        assert serializers.deserialize_uint64(serialized) == node_id

    @pytest.mark.timeout(30)
    def test_serialization_roundtrip(self, gen_graph):
        cg = gen_graph(n_layers=10)
        # Test various node IDs across layers and positions
        for layer in [2, 5, 10]:
            for seg_id in [0, 1, 42, 2**16]:
                node_id = cg.get_node_id(np.uint64(seg_id), layer=layer, x=0, y=0, z=0)
                assert (
                    serializers.deserialize_uint64(
                        serializers.serialize_uint64(node_id)
                    )
                    == node_id
                )

    @pytest.mark.timeout(30)
    def test_serialize_valid_label_id(self):
        label = np.uint64(0x01FF031234556789)
        assert (
            serializers.deserialize_uint64(serializers.serialize_uint64(label)) == label
        )
