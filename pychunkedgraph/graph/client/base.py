import typing
from abc import ABC
from abc import abstractmethod
from datetime import datetime

import numpy as np


class SimpleClient(ABC):
    """
    Abstract class for interacting with backend data store where the chunkedgraph is stored.
    E.g., BigTableClient for using big table as storage.
    """
    
    @abstractmethod
    def create_graph(self) -> None:
        """Initialize the graph and store associated meta."""
    
    @abstractmethod
    def add_graph_version(self, version):
        """Add a version to the graph."""
    
    @abstractmethod
    def read_graph_version(self):
        """Read stored graph version."""
    
    @abstractmethod
    def update_graph_meta(self, meta):
        """Update stored graph meta."""
    
    @abstractmethod
    def read_graph_meta(self):
        """Read stored graph meta."""
    
    @abstractmethod
    def read_nodes(
        self,
        start_id=None,
        end_id=None,
        end_id_inclusive=False,
        user_id=None,
        node_ids=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
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
    def write(
        self,
        rows: typing.Iterable[typing.Dict[str, typing.Union[bytes, typing.Dict[str, typing.Iterable[typing.Any]]]]],
        root_ids: typing.Optional[
            typing.Union[np.uint64, typing.Iterable[np.uint64]]
        ] = None,
        operation_id: typing.Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
        """Writes a list of mutated rows in bulk
        WARNING: If <rows> contains the same row (same row_key) and column
        key two times only the last one is effectively written to the backend data store
        (even when the mutations were applied to different columns)
        --> no versioning!
        :param rows: list
            list of mutated rows
        :param root_ids: list if uint64
        :param operation_id: uint64 or None
            operation_id (or other unique id) that *was* used to lock the root
            the bulk write is only executed if the root is still locked with
            the same id.
        :param slow_retry: bool
        :param block_size: int
        """
    
    @abstractmethod
    def mutate_row(
        self,
        row_key: bytes,
        val_dict: dict,
        time_stamp: typing.Optional[datetime] = None,
    ) -> typing.Dict[str, typing.Union[bytes, typing.Dict[str, typing.Iterable[typing.Any]]]]:
        """Mutates a single row (doesn't write to the backend storage, just returns the row with mutated
        data without writing to the backend storage)."""
    
    @abstractmethod
    def lock_root(
        self,
        root_id,
        operation_id,
    ) -> bool:
        """Attempts to lock the latest version of a root node with operation_id to prevent race conditions."""
    
    @abstractmethod
    def lock_roots(
        self,
        root_ids,
        operation_id,
        future_root_ids_d,
        max_tries: int = 1,
        waittime_s: float = 0.5,
    ) -> typing.Tuple[bool, typing.Iterable]:
        """Locks root nodes to prevent race conditions."""
    
    @abstractmethod
    def lock_root_indefinitely(self, root_id, operation_id):
        """Locks root node with operation_id to prevent race conditions."""
    
    @abstractmethod
    def lock_roots_indefinitely(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: typing.Dict,
    ) -> typing.Tuple[bool, typing.Iterable]:
        """
        Locks root nodes indefinitely to prevent structural damage to graph.
        This scenario is rare and needs asynchronous fix or inspection to unlock.
        """
    
    @abstractmethod
    def unlock_root(self, root_id, operation_id):
        """Unlocks root node that is locked with operation_id."""
    
    @abstractmethod
    def unlock_indefinitely_locked_root(self, root_id, operation_id):
        """Unlocks root node that is indefinitely locked with operation_id."""
    
    @abstractmethod
    def renew_lock(self, root_id, operation_id):
        """Renews existing node lock with operation_id for extended time."""
    
    @abstractmethod
    def renew_locks(self, root_ids, operation_id):
        """Renews existing node locks with operation_id for extended time."""
    
    @abstractmethod
    def get_lock_timestamp(
        self, root_id, operation_id
    ) -> typing.Union[datetime, None]:
        """Reads timestamp from lock row to get a consistent timestamp."""
    
    @abstractmethod
    def get_consolidated_lock_timestamp(self, root_ids, operation_ids):
        """Minimum of multiple lock timestamps."""
    
    @abstractmethod
    def get_compatible_timestamp(self, time_stamp):
        """Datetime time stamp compatible with client's services."""


class ClientWithIDGen(SimpleClient):
    """
    Abstract class for client to backend data store that has support for generating IDs.
    If not, something else can be used but these methods need to be implemented.
    E.g., Big Table row cells can be used to generate unique IDs.
    """
    
    @abstractmethod
    def create_node_ids(self, chunk_id, size):
        """Generate a range of unique IDs in the chunk."""
    
    @abstractmethod
    def create_node_id(
        self, chunk_id: np.uint64, root_chunk=False
    ):
        """Generate a unique ID in the chunk."""
    
    @abstractmethod
    def get_max_node_id(self, chunk_id, root_chunk: bool = False):
        """Gets the current maximum node ID in the chunk."""
    
    @abstractmethod
    def create_operation_id(self):
        """Generate a unique operation ID."""
    
    @abstractmethod
    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""


class OperationLogger(ABC):
    """
    Abstract class for interacting with backend data store where the operation logs are stored.
    E.g., BigTableClient can be used to store logs in Google BigTable.
    """
    
    # TODO add functions for writing
    
    @abstractmethod
    def read_log_entry(self, operation_id: int) -> None:
        """Read log entry for a given operation ID."""
    
    @abstractmethod
    def read_log_entries(self, operation_ids) -> None:
        """Read log entries for given operation IDs."""
