from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Iterable

import numpy as np

from ..meta import ChunkedGraphMeta


class SimpleClient(ABC):
    """
    Abstract class for interacting with backend data store where the chunkedgraph is stored.
    Eg., BigTableClient for using big table as storage.
    """

    @abstractmethod
    def create_graph(self, graph_meta: ChunkedGraphMeta) -> None:
        """Initialize the graph and store associated meta."""

    @abstractmethod
    def read_nodes(
        self,
        start_id=None,
        end_id=None,
        node_ids=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """
        Read nodes and their properties.
        Accepts a range of node IDs or specific node IDs.
        """

    @abstractmethod
    def read_node(
        self,
        node_id: np.uint64,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single node and it's properties."""

    @abstractmethod
    def write_nodes(self, nodes):
        """
        Writes/updates nodes (IDs along with properties).
        Meant to be used when race conditions are not expected.
        Eg., when creating the graph.
        """

    @abstractmethod
    def write_nodes_synchronized(self, nodes, root_ids, operation_id):
        """
        Writes/updates nodes (IDs along with properties)
        by locking root nodes until changes are written.
        """


class ClientWithIDGen(SimpleClient):
    """
    Abstract class for client to backend data store that has support for creating IDs.
    Eg., BigTableClient has locking and concurrency support to generate unique IDs.
    """

    @abstractmethod
    def create_segment_ids(self):
        """Generate a range of unique segment IDs."""

    @abstractmethod
    def create_segment_id(self):
        """Generate a unique segment ID."""

    @abstractmethod
    def get_max_segment_id(self, chunk_id: np.uint64):
        """Gets the current maximum segment ID in the chunk."""

    @abstractmethod
    def create_operation_id(self):
        """Generate a unique operation ID."""

    @abstractmethod
    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""

