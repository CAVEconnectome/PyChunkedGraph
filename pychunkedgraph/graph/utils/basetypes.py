import numpy as np


CHUNK_ID = SEGMENT_ID = NODE_ID = OPERATION_ID = np.dtype("uint64").newbyteorder("L")
EDGE_AFFINITY = np.dtype("float32").newbyteorder("L")
EDGE_AREA = np.dtype("uint64").newbyteorder("L")

COUNTER = np.dtype("int64").newbyteorder("B")

COORDINATES = np.dtype("int64").newbyteorder("L")
CHUNKSIZE = np.dtype("uint64").newbyteorder("L")
FANOUT = np.dtype("uint64").newbyteorder("L")
LAYERCOUNT = np.dtype("uint64").newbyteorder("L")
SPATIALBITS = np.dtype("uint64").newbyteorder("L")
ROOTCOUNTERBITS = np.dtype("uint64").newbyteorder("L")
SKIPCONNECTIONS = np.dtype("uint64").newbyteorder("L")
