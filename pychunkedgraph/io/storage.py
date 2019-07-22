'''
Functions to use when dealing with any cloud storage via CloudVolume
'''

from typing import List

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from .protobuf.chunkEdges_pb3 import Edges

# TODO some funtions in ChunkedGraph
# should be class methods or util functions
# for now pass instance of ChunkedGraph

def get_chunk_edges(cg, chunk_ids: List[np.uint64]):
    fnames = []
    chunk_coords = cg.get_chunk_coordinates(chunk_id)
    chunk_str = '_'.join(str(coord) for coord in chunk_coords)
    fname = f'edges_{chunk_str}.data'

    edgesMessage =  Edges()
    with Storage(cg._cv_path) as st:
        file_content = st.get_files(fnames)

    # TODO move decompression to a generator

    file_content = zstd.ZstdDecompressor().decompressobj().decompress(file_content)
    edgesMessage.ParseFromString(file_content)
    edges = np.frombuffer(edgesMessage.edgeList)
    areas = np.frombuffer(edgesMessage.areas, dtype='<u8')
    affinities = np.frombuffer(edgesMessage.affinities, dtype='<f4')

    return edges, areas, affinities