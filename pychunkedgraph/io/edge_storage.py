'''
Functions to use when dealing with any cloud storage via CloudVolume
'''

import os
from typing import List, Dict

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from .protobuf.chunkEdges_pb2 import Edges


def _decompress_edges(content):
    '''
    :param content: zstd compressed bytes
    :return: Tuple[edges:np.array[np.uint64, np.uint64],
                   areas:np.array[np.uint64]
                   affinities: np.array[np.float64]]
    '''
    edgesMessage =  Edges()
    zstdDecompressorObj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstdDecompressorObj.decompress(content)
    edgesMessage.ParseFromString(file_content)
    
    edges = np.frombuffer(edgesMessage.edgeList).reshape(-1, 2)
    affinities = np.frombuffer(edgesMessage.affinities, dtype='<f4')
    areas = np.frombuffer(edgesMessage.areas, dtype='<u8')
    return edges, affinities, areas


def get_chunk_edges(edges_dir:str, chunks_coordinates: List[np.ndarray], cv_threads):
    '''
    :param: chunks_coordinates np.array of chunk coordinates
    :return: a generator that yields decompressed file content
    '''
    edges_dir = os.environ.get(
        'EDIR', 
        'gs://akhilesh-test/edges/fly_playground/bbox-102_51_5-110_59_9')
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = '_'.join(str(coord) for coord in chunk_coords)
        fnames.append(f'chunk_{chunk_str}_zstd_level_17_proto.data')

    edges = np.array([], dtype=np.uint64).reshape(0, 2)
    affinities = np.array([], dtype=np.float32)
    areas = np.array([], dtype=np.uint64)

    files = []
    with Storage(edges_dir, n_threads = cv_threads) as st:
        files = st.get_files(fnames)
        for _file in files:
            if not _file['content']: continue
            _edges, _affinities, _areas = _decompress_edges(_file['content'])
            edges = np.concatenate([edges, _edges])
            affinities = np.concatenate([affinities, _affinities])
            areas = np.concatenate([areas, _areas])            

    return edges, affinities, areas