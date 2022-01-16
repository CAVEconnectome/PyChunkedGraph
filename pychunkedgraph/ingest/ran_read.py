"""
I made some changes to the segmentation pipeline to reduce the number of small files.
The remap files and the region graph files are merged together into bigger files.
For example, the "in_chunk_xxx_yyy.data" files are merge into a single "in_chunk_xxx.data",
and the reader need to find out the range of interest to extract the data for each chunk.

The layout of the new files is like this:

    byte 1-4: 'SO01' (version identifier)
    byte 5-12: Offset of the index information
    byte 13-20: Length of the index information (including crc32)
    byte 21-n: Payload data of the first chunk
    byte (n+1)-(n+4): Crc32 of the remap data of first chunk
    ...
    ...
    ...
    byte m-l: index data: (chunkid, offset, length)*k
    byte (l+1)-(l+4): Crc32 of the index data

The payload data have the same dtype of the original files, the chunkid is yyy stored in uint64 in this example
For data like "between_chunks_xxx_yyy_zzz.data" it is the pair (yyy,zzz). Here is an example of the reader. (edited)


    real eg:

    [72057594105036800 72057662824513536]
    b'SO01'
    [2664  356]
    (72057594105036811, 72057662824513538, 0., 0, 190.05826, 200, 0., 0)

"""


import numpy as np
from cloudfiles import CloudFiles

cf = CloudFiles(
    "gs://akhilesh-pcg/ranl/rgf/f7ed4db1fb26d94fb40fcebcd7975e3c/agg/chunked_rg/"
)
# fn = "fake_0_0_0_0.data"
fn = "between_chunks_0_4_5_3.data"
# fn = "in_chunk_0_0_0_5.data"

print(fn)

raw = cf[fn]
header = raw[:20]

idx_offset, idx_length = np.frombuffer(header[4:], dtype="uint64")
print("idx_offset, idx_length", idx_offset, idx_length)
print()

idx_length -= 4
idx_content = raw[int(idx_offset) : int(idx_offset + idx_length)]

dt = np.dtype([("chunkid", "2u8"), ("offset", "u8"), ("size", "u8")])
data = np.frombuffer(idx_content, dtype=dt)
print("index_data", data)
print()

chunk = data[0]

print("chunk", chunk)
payload = raw[chunk["offset"] : chunk["offset"] + chunk["size"]]
payload_data = np.frombuffer(payload[:-4], dtype=np.dtype("u8, u8, f4, u8, f4, u8, f4, u8"))
print("payload_data", payload_data)
print()
