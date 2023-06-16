import argparse
from taskqueue import TaskQueue
import os
import pychunkedgraph.meshing.meshing_sqs

if __name__ == '__main__':
    ppid = os.getppid() # Save parent process ID

    def stop_fn_with_parent_health_check():
        if os.getppid() != ppid:
            print("Parent process is gone. {} shutting down...".format(os.getpid()))
            return True
        return False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--qurl', type=str, required=True)
    parser.add_argument('--lease_seconds', type=int, default=100, help='no. of seconds that polling will lease a task before it becomes visible again')
    args = parser.parse_args()
    with TaskQueue(qurl=args.qurl, queue_server='sqs', n_threads=0) as tq:
        tq.poll(stop_fn=stop_fn_with_parent_health_check, lease_seconds=args.lease_seconds)