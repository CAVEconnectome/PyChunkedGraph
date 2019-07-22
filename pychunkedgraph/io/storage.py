'''
Functions to use when dealing with any cloud storage via CloudVolume
'''

from typing import List, Dict

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from .protobuf.chunkEdges_pb3 import Edges


def _decompress_edges(files: List[Dict]):
    '''
    :param files: list of dicts (from CloudVolume.Storage.get_files)
    :return: Tuple[edges:np.array[np.uint64, np.uint64],
                   areas:np.array[np.uint64]
                   affinities: np.array[np.float64]]
    '''
    edgesMessage =  Edges()
    
    for _file in files:
        file_content = zstd.ZstdDecompressor().decompressobj().decompress(_file['content'])
        edgesMessage.ParseFromString(file_content)
        edges = np.frombuffer(edgesMessage.edgeList)
        areas = np.frombuffer(edgesMessage.areas, dtype='<u8')
        affinities = np.frombuffer(edgesMessage.affinities, dtype='<f4')
        yield edges, areas, affinities


def get_chunk_edges(cg, chunk_ids: List[np.uint64]):
    '''
    :param cg: ChunkedGraph instance
    :return: a generator that yields decompressed file content
    '''    
    fnames = []
    for chunk_id in chunk_ids:
        chunk_coords = cg.get_chunk_coordinates(chunk_id)
        chunk_str = '_'.join(str(coord) for coord in chunk_coords)
        fnames.append(f'edges_{chunk_str}.data')

    files = []
    with Storage(f'{cg._cv_path}/edges_dir') as st:
        files = st.get_files(fnames)

    return _decompress_edges(files)