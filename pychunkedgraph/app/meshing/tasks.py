from pychunkedgraph.app import app_utils
from pychunkedgraph.meshing import meshgen, meshgen_utils
import numpy as np
import os
import redis
from rq import Queue, Connection, Retry
from rq_manager import manager
from flask import current_app
import collections

def remeshing(table_id, lvl2_nodes):
    lvl2_nodes = np.array(lvl2_nodes, dtype=np.uint64)
    cg = app_utils.get_cg(table_id, skip_cache=True)

    cv_mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    cv_unsharded_mesh_path = os.path.join(
        cg.meta.data_source.WATERSHED, cv_mesh_dir, cv_unsharded_mesh_dir
    )
    mesh_data = cg.meta.custom_data["mesh"]

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg,
        lvl2_nodes,
        stop_layer=mesh_data["max_layer"],
        mip=mesh_data["mip"],
        max_err=mesh_data["max_error"],
        cv_sharded_mesh_dir=cv_mesh_dir,
        cv_unsharded_mesh_path=cv_unsharded_mesh_path,
    )

def remeshing_v2(table_id, lvl2_nodes, is_priority):
    lvl2_nodes = np.array(lvl2_nodes, dtype=np.uint64)
    cg = app_utils.get_cg(table_id, skip_cache=False)

    cv_mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    cv_unsharded_mesh_path = os.path.join(
        cg.meta.data_source.WATERSHED, cv_mesh_dir, cv_unsharded_mesh_dir
    )
    mesh_data = cg.meta.custom_data["mesh"]

    with Connection(redis.from_url(current_app.config["REDIS_URL"])):
        if is_priority:
            retry=Retry(max=3, interval=[1, 10, 60])
            queue_name = "mesh-chunks"
        else:
            retry=Retry(max=3, interval=[60, 60, 60])
            queue_name = "mesh-chunks-low-priority"
        q = Queue(queue_name, retry=retry, default_timeout=1200)

        l2_chunk_dict = collections.defaultdict(set)
        # Find the chunk_ids of the l2_node_ids

        def add_nodes_to_l2_chunk_dict(ids):
            for node_id in ids:
                chunk_id = cg.get_chunk_id(node_id)
                l2_chunk_dict[chunk_id].add(node_id)

        add_nodes_to_l2_chunk_dict(lvl2_nodes)
        
        lvl_2_jobs = []
        for chunk_id, node_ids in l2_chunk_dict.items():
            job = {
                  'func': mesh_lvl2_node,
                  'args': (table_id,
                             chunk_id,
                             mesh_data["mip"],
                             node_ids,
                             cv_unsharded_mesh_path,
                             mesh_data["max_error"])
                 }
            lvl_2_jobs.append(job)
        project = {
            'jobs': [{
                'blocking': True,
                'jobs': lvl_2_jobs
            }]
        }

        stop_layer = mesh_data["max_layer"]
        chunk_dicts = []
        max_layer = stop_layer or cg._n_layers
        for layer in range(3, max_layer + 1):
            chunk_dicts.append(collections.defaultdict(set))
        cur_chunk_dict = l2_chunk_dict
        # Find the parents of each l2_node_id up to the stop_layer, as well as their associated chunk_ids
        for layer in range(3, max_layer + 1):
            
            for _, node_ids in cur_chunk_dict.items():
                parent_nodes = cg.get_parents(node_ids)
                for parent_node in parent_nodes:
                    chunk_layer = cg.get_chunk_layer(parent_node)
                    index_in_dict_array = chunk_layer - 3
                    if index_in_dict_array < len(chunk_dicts):
                        chunk_id = cg.get_chunk_id(parent_node)
                        chunk_dicts[index_in_dict_array][chunk_id].add(parent_node)
            cur_chunk_dict = chunk_dicts[layer - 3]

        for chunk_dict in chunk_dicts:
            layer_subjobs = []
            for chunk_id, node_ids in chunk_dict.items():
                job = {'func': stitch_chunks,
                       'args': (table_id,
                                chunk_id,
                                mesh_data["mip"],
                                40,
                                node_ids,
                                cv_mesh_dir,
                                cv_unsharded_mesh_path)
                }
                layer_subjobs.append(job)
            layer_job = {
                'blocking': True,
                'jobs': layer_subjobs
            }
            project['jobs'].append(layer_job)

        task = q.enqueue(manager, project)

def mesh_lvl2_node(
    table_id,
    chunk_id,
    mip,
    node_ids,
    cv_unsharded_mesh_path,
    max_err):

    cg = app_utils.get_cg(table_id, skip_cache=True)
    meshgen.chunk_initial_unsharded_mesh_task(
            None,
            chunk_id,
            mip=mip,
            node_id_subset=node_ids,
            cg=cg,
            cv_unsharded_mesh_path=cv_unsharded_mesh_path,
            max_err=max_err,
    )
    
def stitch_chunks(table_id,
                  chunk_id,
                  mip,
                  fragment_batch_size,
                  node_ids,
                  cv_sharded_mesh_dir,
                  cv_unsharded_mesh_path):

    cg = app_utils.get_cg(table_id, skip_cache=True)
    meshgen.chunk_stitch_remeshing_task(
                None,
                chunk_id,
                mip=mip,
                fragment_batch_size=fragment_batch_size,
                node_id_subset=node_ids,
                cg=cg,
                cv_sharded_mesh_dir=cv_sharded_mesh_dir,
                cv_unsharded_mesh_path=cv_unsharded_mesh_path)