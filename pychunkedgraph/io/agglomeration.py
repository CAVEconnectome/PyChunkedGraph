import json

from cloudvolume.storage import SimpleStorage


def put_chunk_agglomeration(agglomeration_dir, mapping, chunk_coord):
    # filename format - chunk_x_y_z.serliazation
    file_name = f"chunk_{'_'.join(str(coord) for coord in chunk_coord)}.json"
    with SimpleStorage(agglomeration_dir) as storage:
        storage.put_file(
            file_path=file_name,
            content=json.dumps(mapping).encode("utf-8"),
            compress="gzip",
            cache_control="no-cache",
        )

def get_chunk_agglomeration(agglomeration_dir, chunk_coord):
    file_name = f"chunk_{'_'.join(str(coord) for coord in chunk_coord)}.json"
    with SimpleStorage(agglomeration_dir) as storage:
        content = storage.get_file(file_name)
        return json.loads(content.decode('utf-8'))

