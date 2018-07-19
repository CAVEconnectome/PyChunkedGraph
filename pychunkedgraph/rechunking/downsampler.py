

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


def downsample_dataset(dataset_name, from_mip=-1, num_mips=1, n_threads=32):
    if dataset_name == "pinky":
        ws_path = "gs://neuroglancer/svenmd/pinky40_v11/watershed/"
    elif dataset_name == "basil":
        ws_path = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed/"
    else:
        raise Exception("Dataset unknown")

    with LocalTaskQueue(parallel=n_threads) as task_queue:
        tc.create_downsampling_tasks(task_queue, ws_path, mip=from_mip,
                                     fill_missing=True, num_mips=num_mips,
                                     preserve_chunk_size=True)


def downsample_dataset_multiple_mips_memory_efficient(dataset_name, num_mips,
                                                      n_threads=32):
    for i_mip in range(0, num_mips, 3):
        print("\n\nMIP %d\n\n" % (i_mip + 1))
        downsample_dataset(dataset_name, from_mip=i_mip, num_mips=3,
                           n_threads=n_threads)
