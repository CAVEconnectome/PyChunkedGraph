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
    def read_byte_rows(
        self,
        start_key: Optional[bytes] = None,
        end_key: Optional[bytes] = None,
        row_keys: Optional[Iterable[bytes]] = None,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Dict[
        bytes,
        Union[
            Dict[column_keys._Column, List[bigtable.row_data.Cell]],
            List[bigtable.row_data.Cell],
        ],
    ]:


    @abstractmethod
    def read_byte_row(
        self,
        row_key: bytes,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Union[
        Dict[column_keys._Column, List[bigtable.row_data.Cell]],
        List[bigtable.row_data.Cell],
    ]:

    @abstractmethod
    def read_nodes(
        self,
        start_id: Optional[np.uint64] = None,
        end_id: Optional[np.uint64] = None,
        node_ids: Optional[Iterable[np.uint64]] = None,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Dict[np.uint64,Union[Dict,List]]:


    @abstractmethod
    def read_node_id_row(
        self,
        node_id: np.uint64,
        columns: Optional[
            Union[Iterable[column_keys._Column], column_keys._Column]
        ] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        end_time_inclusive: bool = False,
    ) -> Union[
        Dict[column_keys._Column, List[bigtable.row_data.Cell]],
        List[bigtable.row_data.Cell],
    ]:

