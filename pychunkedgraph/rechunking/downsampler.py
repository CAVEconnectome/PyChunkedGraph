from taskqueue import LocalTaskQueue, TaskQueue
import igneous.task_creation as tc


def downsample_dataset(dataset_name, from_mip=-1, num_mips=1, local=False,
                       n_threads=32):
    if dataset_name == "pinky":
        ws_path = "gs://neuroglancer/svenmd/pinky40_v11/watershed/"
    elif dataset_name == "basil":
        ws_path = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed/"
    elif dataset_name == "pinky100":
        ws_path = "gs://neuroglancer/nkem/pinky100_v0/ws/lost_no-random/bbox1_0/"
    else:
        raise Exception("Dataset unknown")

    if local:
        with LocalTaskQueue(parallel=n_threads) as task_queue:
            tc.create_downsampling_tasks(task_queue, ws_path, mip=from_mip,
                                         fill_missing=True, num_mips=num_mips,
                                         preserve_chunk_size=True)
    else:
        with TaskQueue(queue_server='sqs',
                       qurl="https://sqs.us-east-1.amazonaws.com/098703261575/nkem-igneous") as task_queue:
            tc.create_downsampling_tasks(task_queue, ws_path, mip=from_mip,
                                         fill_missing=True, num_mips=num_mips,
                                         preserve_chunk_size=True)


def downsample_dataset_multiple_mips_memory_efficient(dataset_name, num_mips,
                                                      n_threads=32):
    for i_mip in range(0, num_mips, 3):
        print("\n\nMIP %d\n\n" % (i_mip + 1))
        downsample_dataset(dataset_name, from_mip=i_mip, num_mips=3,
                           n_threads=n_threads)
