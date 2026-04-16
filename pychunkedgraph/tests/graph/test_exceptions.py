"""Tests for pychunkedgraph.graph.exceptions"""

from pychunkedgraph.graph.exceptions import (
    SupervoxelSplitRequiredError,
    ChunkedGraphError,
)


class TestSupervoxelSplitRequiredError:
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
        try:
            raise SupervoxelSplitRequiredError("test", {1: 2})
        except ChunkedGraphError as e:
            assert e.sv_remapping == {1: 2}
