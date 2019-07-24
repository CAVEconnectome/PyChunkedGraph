'''
Functions to use when dealing with any cloud storage via CloudVolume
'''

import os
from typing import List, Dict

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from .protobuf.chunkEdges_pb2 import Edges


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
        affinities = np.frombuffer(edgesMessage.affinities, dtype='<f4')
        areas = np.frombuffer(edgesMessage.areas, dtype='<u8')
        yield edges, affinities, areas


def get_chunk_edges(edges_dir:str, chunks_coordinates: List[np.ndarray]):
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
        fnames.append(f'edges_{chunk_str}.data')

    files = []
    with Storage(edges_dir) as st:
        files = st.get_files(fnames)

    chunks_edges = _decompress_edges(files)

    edges = np.array([], dtype=np.uint64).reshape(0, 2)
    affinities = np.array([], dtype=np.float32)
    areas = np.array([], dtype=np.uint64)

    for chunk_edges in chunks_edges:
        _edges, _affinities, _areas = chunk_edges
        areas = np.concatenate([areas, _areas])
        affinities = np.concatenate([affinities, _affinities])
        edges = np.concatenate([edges, _edges])

    return edges, affinities, areas