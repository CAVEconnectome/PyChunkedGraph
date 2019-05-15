import numpy as np


from pychunkedgraph.ingest import ran_ingestion as ri
from pychunkedgraph.benchmarking import graph_measurements as gm, timings


def create_benchmark_datasets(storage_path,
                              ws_cv_path,
                              cg_table_base_id,
                              chunk_size=[256, 256, 512],
                              use_skip_connections=True,
                              s_bits_atomic_layer=None,
                              fan_out=2,
                              aff_dtype=np.float32,
                              start_iter=0,
                              n_iter=8,
                              run_gm=False,
                              run_bm=False,
                              job_size=250,
                              instance_id=None,
                              project_id=None,
                              n_threads=[64, 64]):

    chunk_size = np.array(chunk_size)

    for i_iter in range(start_iter, n_iter):
        size = chunk_size * 2 ** (i_iter + 1)
        cg_table_id = f"{cg_table_base_id}_s{i_iter}"

        ri.ingest_into_chunkedgraph(storage_path=storage_path,
                                    ws_cv_path=ws_cv_path,
                                    cg_table_id=cg_table_id,
                                    chunk_size=chunk_size,
                                    use_skip_connections=use_skip_connections,
                                    s_bits_atomic_layer=s_bits_atomic_layer,
                                    fan_out=fan_out,
                                    aff_dtype=aff_dtype,
                                    size=size,
                                    instance_id=instance_id,
                                    project_id=project_id,
                                    start_layer=1,
                                    n_threads=n_threads)

        if run_gm or run_bm:
            gm.run_graph_measurements(table_id=cg_table_id, n_threads=n_threads[0])

            if run_bm:
                timings.run_timings(table_id=cg_table_id, job_size=job_size)


def compute_graph_measurements_dataset(cg_table_base_id, start_iter=0, n_iter=8,
                                       n_threads=1):
    for i_iter in range(start_iter, n_iter):
        cg_table_id = f"{cg_table_base_id}_s{i_iter}"
        gm.run_graph_measurements(table_id=cg_table_id, n_threads=n_threads)


def compute_benchmarks_dataset(cg_table_base_id, start_iter=0, n_iter=8,
                               job_size=500):
    for i_iter in range(start_iter, n_iter):
        cg_table_id = f"{cg_table_base_id}_s{i_iter}"
        timings.run_timings(table_id=cg_table_id, job_size=job_size)

