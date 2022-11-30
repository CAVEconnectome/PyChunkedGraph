import sys
from rq import Connection, Worker

# Preload libraries from pychunkedgraph.ingest.cluster
from typing import Sequence, Tuple

import numpy as np

from pychunkedgraph.ingest.utils import chunk_id_str
from pychunkedgraph.ingest.manager import IngestionManager
from pychunkedgraph.ingest.common import get_atomic_chunk_data
from pychunkedgraph.ingest.ran_agglomeration import get_active_edges
from pychunkedgraph.ingest.create.atomic_layer import add_atomic_edges
from pychunkedgraph.ingest.create.abstract_layers import add_layer
from pychunkedgraph.graph.meta import ChunkedGraphMeta
from pychunkedgraph.graph.chunks.hierarchy import get_children_chunk_coords
from pychunkedgraph.utils.redis import keys as r_keys
from pychunkedgraph.utils.redis import get_redis_connection

qs = sys.argv[1:]
w = Worker(qs, connection=get_redis_connection())
w.work()