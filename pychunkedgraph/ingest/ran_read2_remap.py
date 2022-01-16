import numpy as np
from cloudfiles import CloudFiles

cf = CloudFiles(
    "gs://akhilesh-pcg/ranl/rgf/f7ed4db1fb26d94fb40fcebcd7975e3c/agg/remap/"
)
fn = "done_0_0_6_1.data"
# fn = "between_chunks_0_0_1_2.data"
# fn = "in_chunk_0_0_0_5.data"

print(fn)

raw = cf[fn]
header = raw[:20]

idx_offset, idx_length = np.frombuffer(header[4:], dtype="uint64")
print("idx_offset, idx_length", idx_offset, idx_length)
print()

idx_length -= 4
idx_content = raw[int(idx_offset) : int(idx_offset + idx_length)]

dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
index_data = np.frombuffer(idx_content, dtype=dt)
print("index_data", index_data)
print()

for d in index_data:
    chunk = d

print("chunk", chunk)
payload = raw[chunk["offset"] : chunk["offset"] + chunk["size"]]
payload_data = np.frombuffer(payload[:-4], dtype=np.uint64)
print("payload_data", payload_data)
print()
