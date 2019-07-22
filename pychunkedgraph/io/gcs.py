'''
Functions to use when dealing with Google Cloud Storage
'''

# TODO some funtions in ChunkedGraph
# should be class methods or util functions
# for now pass instance of ChunkedGraph

def get_chunk_edges(cg, chunk_id):
    chunk_coords = cg.get_chunk_coordinates(chunk_id)
    chunk_str = repr(chunk)[1:-1].replace(', ','_')
    fname = f'edges_{chunk_str}.data'
