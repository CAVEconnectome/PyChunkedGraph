from abc import ABC
from abc import abstractmethod

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
    def write_nodes(self, nodes):
        """Writes/updates nodes (IDs along with properties)."""

    @abstractmethod
    def lock_nodes(self, node_ids):
        """Locks nodes to prevent race conditions."""

    @abstractmethod
    def lock_node(self, node_id, operation_id):
        """Locks node with operation_id to prevent race conditions."""


class ClientWithIDGen(SimpleClient):
    """
    Abstract class for client to backend data store that has support for generating IDs.
    Eg., BigTableClient has locking and concurrency support to generate unique IDs.
    """

    @abstractmethod
    def create_node_ids(self, chunk_id: np.uint64):
        """Generate a range of unique node IDs."""

    @abstractmethod
    def create_node_id(self, chunk_id: np.uint64):
        """Generate a unique node ID."""

    @abstractmethod
    def get_max_node_id(self, chunk_id: np.uint64):
        """Gets the current maximum node ID in the chunk."""

    @abstractmethod
    def create_operation_id(self):
        """Generate a unique operation ID."""

    @abstractmethod
    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""


class ClientWithLogging(SimpleClient):
    """Abstract class for client to backend data store with support for logging."""

    @abstractmethod
    def read_logs(self, operation_ids=None):
        """
        Read log entries of given operation IDs.
        If None, reads all log entires.
        """

