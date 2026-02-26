"""Tests for pychunkedgraph.graph.ocdbt"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestGetSegSourceAndDestinationOcdbt:
    @patch("pychunkedgraph.graph.ocdbt.ts")
    def test_returns_src_dst_tuple(self, mock_ts):
        from pychunkedgraph.graph.ocdbt import get_seg_source_and_destination_ocdbt

        mock_src = MagicMock()
        mock_schema = MagicMock()
        mock_schema.rank = 4
        mock_schema.dtype = "uint64"
        mock_schema.codec = None
        mock_schema.domain = None
        mock_schema.shape = [256, 256, 256, 1]
        mock_schema.chunk_layout = None
        mock_schema.dimension_units = None
        mock_src.schema = mock_schema

        mock_dst = MagicMock()

        # ts.open returns a future-like with .result()
        mock_ts.open.side_effect = [
            MagicMock(result=MagicMock(return_value=mock_src)),
            MagicMock(result=MagicMock(return_value=mock_dst)),
        ]

        src, dst = get_seg_source_and_destination_ocdbt("gs://bucket/ws")
        assert src is mock_src
        assert dst is mock_dst
        assert mock_ts.open.call_count == 2

    @patch("pychunkedgraph.graph.ocdbt.ts")
    def test_create_flag(self, mock_ts):
        from pychunkedgraph.graph.ocdbt import get_seg_source_and_destination_ocdbt

        mock_src = MagicMock()
        mock_schema = MagicMock()
        mock_schema.rank = 4
        mock_schema.dtype = "uint64"
        mock_schema.codec = None
        mock_schema.domain = None
        mock_schema.shape = [256, 256, 256, 1]
        mock_schema.chunk_layout = None
        mock_schema.dimension_units = None
        mock_src.schema = mock_schema

        mock_dst = MagicMock()
        mock_ts.open.side_effect = [
            MagicMock(result=MagicMock(return_value=mock_src)),
            MagicMock(result=MagicMock(return_value=mock_dst)),
        ]

        src, dst = get_seg_source_and_destination_ocdbt("gs://bucket/ws", create=True)

        # Second ts.open call should have create=True and delete_existing=True
        second_call = mock_ts.open.call_args_list[1]
        assert second_call.kwargs.get("create") == True
        assert second_call.kwargs.get("delete_existing") == True


class TestCopyWsChunk:
    def test_basic_copy(self):
        from pychunkedgraph.graph.ocdbt import copy_ws_chunk

        mock_source = MagicMock()
        mock_destination = MagicMock()

        # Simulate source read
        data = np.ones((64, 64, 64), dtype=np.uint64)
        mock_source.__getitem__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=data))
                )
            )
        )
        mock_destination.__getitem__ = MagicMock(
            return_value=MagicMock(
                write=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=None))
                )
            )
        )

        voxel_bounds = np.array([[0, 256], [0, 256], [0, 256]])
        copy_ws_chunk(
            mock_source,
            mock_destination,
            chunk_size=(64, 64, 64),
            coords=[0, 0, 0],
            voxel_bounds=voxel_bounds,
        )
        # Should have read from source and written to destination
        mock_source.__getitem__.assert_called_once()
        mock_destination.__getitem__.assert_called_once()

    def test_boundary_clipping(self):
        from pychunkedgraph.graph.ocdbt import copy_ws_chunk

        mock_source = MagicMock()
        mock_destination = MagicMock()

        data = np.ones((32, 64, 64), dtype=np.uint64)
        mock_source.__getitem__ = MagicMock(
            return_value=MagicMock(
                read=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=data))
                )
            )
        )
        mock_destination.__getitem__ = MagicMock(
            return_value=MagicMock(
                write=MagicMock(
                    return_value=MagicMock(result=MagicMock(return_value=None))
                )
            )
        )

        # Volume ends at 224 in x, so last chunk (192-256) is clipped to (192-224)
        voxel_bounds = np.array([[0, 224], [0, 256], [0, 256]])
        copy_ws_chunk(
            mock_source,
            mock_destination,
            chunk_size=(64, 64, 64),
            coords=[3, 0, 0],
            voxel_bounds=voxel_bounds,
        )
        mock_source.__getitem__.assert_called_once()


class TestOcdbtConstants:
    def test_compression_level(self):
        from pychunkedgraph.graph.ocdbt import OCDBT_SEG_COMPRESSION_LEVEL

        assert OCDBT_SEG_COMPRESSION_LEVEL == 17
        assert isinstance(OCDBT_SEG_COMPRESSION_LEVEL, int)
