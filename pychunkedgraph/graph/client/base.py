from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Iterable

import numpy as np

from ..meta import ChunkedGraphMeta


# TODO design api
# 1. create / overwrite
# 2. a counter to generate unique ids (IDs api?)
# 3. store metadata
# 4. read/write rows api


class Client(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def create_graph(self, graph_meta: ChunkedGraphMeta) -> None:
        """Create graph and store associated meta"""

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
        """Read nodes and their properties."""

    @abstractmethod
    def read_node(
        self,
        node_id: np.uint64,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single node and it's properties"""
