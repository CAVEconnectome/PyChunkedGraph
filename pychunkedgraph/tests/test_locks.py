from time import sleep
from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from .helpers import create_chunk, to_label
from ..graph.lineage import get_future_root_ids
from ..ingest.create.parent_layer import add_parent_chunk


class TestGraphLocks:
    @pytest.mark.timeout(30)
    def test_lock_unlock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try unlock (opid = 1)
        (4) Try lock (opid = 2)
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
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

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_root(root_id=root_id, operation_id=operation_id_1)

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    @pytest.mark.timeout(30)
    def test_lock_expiration(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """
        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
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

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds())

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
            max_tries=10,
            waittime_s=0.5,
        )[0]

    @pytest.mark.timeout(30)
    def test_lock_renew(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try lock (opid = 1)
        (2) Try lock (opid = 2)
        (3) Try lock (opid = 2) with retries
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
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

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.renew_locks(root_ids=[root_id], operation_id=operation_id_1)

    @pytest.mark.timeout(30)
    def test_lock_merge_lock_old_id(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Merge (includes lock opid 1)
        (2) Try lock opid 2 --> should be successful and return new root id
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
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

        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        new_root_ids = cg.add_edges(
            "Chuck Norris",
            [to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            affinities=1.0,
        ).new_root_ids

        assert new_root_ids is not None

        operation_id_2 = cg.id_client.create_operation_id()
        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        success, new_root_id = cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
            max_tries=10,
            waittime_s=0.5,
        )

        assert success
        assert new_root_ids[0] == new_root_id

    @pytest.mark.timeout(30)
    def test_indefinite_lock(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try indefinite lock (opid = 1), get indefinite lock
        (2) Try normal lock (opid = 2), doesn't get the normal lock
        (3) Try unlock indefinite lock (opid = 1), should unlock indefinite lock
        (4) Try lock (opid = 2), should get the normal lock
        """

        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
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

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}
        assert cg.client.lock_roots_indefinitely(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_indefinitely_locked_root(
            root_id=root_id, operation_id=operation_id_1
        )

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

    @pytest.mark.timeout(30)
    def test_indefinite_lock_with_normal_lock_expiration(self, gen_graph):
        """
        No connection between 1, 2 and 3
        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1  │  3  │
        │  2  │     │
        └─────┴─────┘

        (1) Try normal lock (opid = 1), get normal lock
        (2) Try indefinite lock (opid = 1), get indefinite lock
        (3) Wait until normal lock expires
        (4) Try normal lock (opid = 2), doesn't get the normal lock
        (5) Try unlock indefinite lock (opid = 1), should unlock indefinite lock
        (6) Try lock (opid = 2), should get the normal lock
        """

        # 1. TODO renew lock test when getting indefinite lock
        cg = gen_graph(n_layers=3)

        # Preparation: Build Chunk A
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 1), to_label(cg, 1, 0, 0, 0, 2)],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Preparation: Build Chunk B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 1)],
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

        operation_id_1 = cg.id_client.create_operation_id()
        root_id = cg.get_root(to_label(cg, 1, 0, 0, 0, 1))

        future_root_ids_d = {root_id: get_future_root_ids(cg, root_id)}

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.lock_roots_indefinitely(
            root_ids=[root_id],
            operation_id=operation_id_1,
            future_root_ids_d=future_root_ids_d,
        )[0]

        sleep(cg.meta.graph_config.ROOT_LOCK_EXPIRY.total_seconds())

        operation_id_2 = cg.id_client.create_operation_id()
        assert not cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]

        assert cg.client.unlock_indefinitely_locked_root(
            root_id=root_id, operation_id=operation_id_1
        )

        assert cg.client.lock_roots(
            root_ids=[root_id],
            operation_id=operation_id_2,
            future_root_ids_d=future_root_ids_d,
        )[0]
