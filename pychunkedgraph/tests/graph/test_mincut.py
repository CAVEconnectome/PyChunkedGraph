from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from ..helpers import create_chunk, to_label
from ...graph import exceptions
from ...ingest.create.parent_layer import add_parent_chunk


class TestGraphMinCut:
    # TODO: Ideally, those tests should focus only on mincut retrieving the correct edges.
    #       The edge removal part should be tested exhaustively in TestGraphSplit
    @pytest.mark.timeout(30)
    def test_cut_regular_link(self, gen_graph):
        """
        Regular link between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        # Mincut
        new_root_ids = cg.remove_edges(
            "Jane Doe",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            source_coords=[0, 0, 0],
            sink_coords=[
                2 * cg.meta.graph_config.CHUNK_SIZE[0],
                2 * cg.meta.graph_config.CHUNK_SIZE[1],
                cg.meta.graph_config.CHUNK_SIZE[2],
            ],
            mincut=True,
            disallow_isolating_cut=True,
        ).new_root_ids

        # verify new state
        assert len(new_root_ids) == 2
        assert cg.get_root(to_label(cg, 1, 0, 0, 0, 0)) != cg.get_root(
            to_label(cg, 1, 1, 0, 0, 0)
        )
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(to_label(cg, 1, 0, 0, 0, 0))], leaves_only=True
            )
        )
        assert len(leaves) == 1 and to_label(cg, 1, 0, 0, 0, 0) in leaves
        leaves = np.unique(
            cg.get_subgraph(
                [cg.get_root(to_label(cg, 1, 1, 0, 0, 0))], leaves_only=True
            )
        )
        assert len(leaves) == 1 and to_label(cg, 1, 1, 0, 0, 0) in leaves

    @pytest.mark.timeout(30)
    def test_cut_no_link(self, gen_graph):
        """
        No connection between 1 and 2
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 0),
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        res_new = cg.client.read_all_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_cut_old_link(self, gen_graph):
        """
        Link between 1 and 2 got removed previously (aff = 0.0)
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1┅┅╎┅┅2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )
        cg.remove_edges(
            "John Doe",
            source_ids=to_label(cg, 1, 1, 0, 0, 0),
            sink_ids=to_label(cg, 1, 0, 0, 0, 0),
            mincut=False,
        )

        res_old = cg.client.read_all_rows()
        res_old.consume_all()

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 0),
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        res_new = cg.client.read_all_rows()
        res_new.consume_all()

        assert res_new.rows == res_old.rows

    @pytest.mark.timeout(30)
    def test_cut_indivisible_link(self, gen_graph):
        """
        Sink: 1, Source: 2
        Link between 1 and 2 is set to `inf` and must not be cut.
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1══╪══2  │
        │     │     │
        └─────┴─────┘
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), inf)],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), inf)],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(
            cg,
            3,
            [0, 0, 0],
            time_stamp=fake_timestamp,
            n_threads=1,
        )

        original_parents_1 = cg.get_root(
            to_label(cg, 1, 0, 0, 0, 0), get_all_parents=True
        )
        original_parents_2 = cg.get_root(
            to_label(cg, 1, 1, 0, 0, 0), get_all_parents=True
        )

        # Mincut
        with pytest.raises(exceptions.PostconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=to_label(cg, 1, 0, 0, 0, 0),
                sink_ids=to_label(cg, 1, 1, 0, 0, 0),
                source_coords=[0, 0, 0],
                sink_coords=[
                    2 * cg.meta.graph_config.CHUNK_SIZE[0],
                    2 * cg.meta.graph_config.CHUNK_SIZE[1],
                    cg.meta.graph_config.CHUNK_SIZE[2],
                ],
                mincut=True,
            )

        new_parents_1 = cg.get_root(to_label(cg, 1, 0, 0, 0, 0), get_all_parents=True)
        new_parents_2 = cg.get_root(to_label(cg, 1, 1, 0, 0, 0), get_all_parents=True)

        assert np.all(np.array(original_parents_1) == np.array(new_parents_1))
        assert np.all(np.array(original_parents_2) == np.array(new_parents_2))

    @pytest.mark.timeout(30)
    def test_mincut_disrespects_sources_or_sinks(self, gen_graph):
        """
        When the mincut separates sources or sinks, an error should be thrown.
        Although the mincut is setup to never cut an edge between two sources or
        two sinks, this can happen when an edge along the only path between two
        sources or two sinks is cut.
        """
        cg = gen_graph(n_layers=2)

        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[
                to_label(cg, 1, 0, 0, 0, 0),
                to_label(cg, 1, 0, 0, 0, 1),
                to_label(cg, 1, 0, 0, 0, 2),
                to_label(cg, 1, 0, 0, 0, 3),
            ],
            edges=[
                (to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 2), 2),
                (to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2), 3),
                (to_label(cg, 1, 0, 0, 0, 2), to_label(cg, 1, 0, 0, 0, 3), 10),
            ],
            timestamp=fake_timestamp,
        )

        # Mincut
        with pytest.raises(exceptions.PreconditionError):
            cg.remove_edges(
                "Jane Doe",
                source_ids=[to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 1)],
                sink_ids=[to_label(cg, 1, 0, 0, 0, 3)],
                source_coords=[[0, 0, 0], [10, 0, 0]],
                sink_coords=[[5, 5, 0]],
                mincut=True,
            )
