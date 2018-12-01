#!/usr/bin/env python3

from chunkedgraph.meshing.meshgen import mesh_lvl2_preview
import chunkedgraph
import os
from celery import Celery
from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)

# set username and password for broker, with overrides from environment variables
rabbitmq_user = os.environ.get('RABBITMQ_USERNAME', 'rabbit')
rabbitmq_password = os.environ.get('RABBITMQ_PASSWORD', 'bitnami')
rabbitmq_vhost = os.environ.get('RABBITMQ_VHOST', '/')

# Get Kubernetes-provided address of the broker service
broker_service_host = os.environ.get(
    'MESSAGE_QUEUE_SERVICE_HOST', ' rabbitmq-daf-rabbitmq-svc')
# could also use DNS name: 'message-queue'|'message-queue.default.svc.cluster.local'
# for our default

broker_url = 'amqp://%s:%s@%s/%s' % (rabbitmq_user,
                                     rabbitmq_password,
                                     broker_service_host,
                                     rabbitmq_vhost)
app = Celery('tasks', broker=broker_url, backend='amqp')


@app.task
def mesh_lvl2_previews_threads(serialized_cg_info, lvl2_node_id,
                               supervoxel_ids,
                               cv_path, cv_mesh_dir, mip,
                               simplification_factor,
                               max_err, parallel_download, verbose,
                               cache_control):

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)
    mesh_lvl2_preview(cg, lvl2_node_id, supervoxel_ids=supervoxel_ids,
                      cv_path=cv_path, cv_mesh_dir=cv_mesh_dir, mip=mip,
                      simplification_factor=simplification_factor,
                      max_err=max_err, parallel_download=parallel_download,
                      verbose=verbose, cache_control=cache_control)
