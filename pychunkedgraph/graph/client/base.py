from abc import ABC
from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from ..meta import ChunkedGraphMeta


class SimpleClient(ABC):
    """
    Abstract class for interacting with backend data store where the chunkedgraph is stored.
    Eg., BigTableClient for using big table as storage.
    """

    @abstractmethod
    def create_graph(self) -> None:
        """Initialize the graph and store associated meta."""

    @abstractmethod
    def update_graph_meta(self, meta):
        """Update stored graph meta."""

    @abstractmethod
    def read_graph_meta(self):
        """Read stored graph meta."""

    @abstractmethod
    def update_graph_provenance(self, provenance):
        """Update how the graph was created."""

    @abstractmethod
    def read_graph_provenance(self):
        """Read how the graph was created."""

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
        node_id,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single node and it's properties."""

    @abstractmethod
    def write_nodes(self, nodes):
        """Writes/updates nodes (IDs along with properties)."""

    @abstractmethod
    def lock_root(self, node_id, operation_id):
        """Locks root node with operation_id to prevent race conditions."""

    @abstractmethod
    def lock_roots(self, node_ids):
        """Locks root nodes to prevent race conditions."""

    @abstractmethod
    def unlock_root(self, node_id, operation_id):
        """Unlocks root node that is locked with operation_id."""

    @abstractmethod
    def renew_lock(self, node_id, operation_id):
        """Renews existing node lock with operation_id for extended time."""

    @abstractmethod
    def renew_locks(self, node_ids, operation_id):
        """Renews existing node locks with operation_id for extended time."""


class ClientWithIDGen(SimpleClient):
    """
    Abstract class for client to backend data store that has support for generating IDs.
    If not, something else can be used but these methods need to be implemented.
    Eg., Big Table row cells can be used to generate unique IDs.
    """

    @abstractmethod
    def create_node_ids(self, chunk_id: np.uint64):
        """Generate a range of unique IDs in the chunk."""

    @abstractmethod
    def create_node_id(self, chunk_id: np.uint64):
        """Generate a unique ID in the chunk."""

    @abstractmethod
    def get_max_node_id(self, chunk_id: np.uint64):
        """Gets the current maximum node ID in the chunk."""

    @abstractmethod
    def create_operation_id(self):
        """Generate a unique operation ID."""

    @abstractmethod
    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""
