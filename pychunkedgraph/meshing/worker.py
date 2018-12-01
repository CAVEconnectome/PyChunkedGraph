#!/usr/bin/env python3

from chunkedgraph.meshing.meshgen import mesh_lvl2_preview
import chunkedgraph
import os
from celery import Celery
from celery.utils.log import get_task_logger
import pika
import json
logger = get_task_logger(__name__)

# set username and password for broker, with overrides from environment variables
rabbitmq_user = os.environ.get('RABBITMQ_USERNAME', 'rabbit')
rabbitmq_password = os.environ.get('RABBITMQ_PASSWORD', 'bitnami')
rabbitmq_vhost = os.environ.get('RABBITMQ_VHOST', '/')
remesh_exchange = os.environ.get('RABBIGMQ_REMESH_EXCHANGE', 'remeshing')
# Get Kubernetes-provided address of the broker service
broker_service_host = os.environ.get(
    'MESSAGE_QUEUE_SERVICE_HOST', ' rabbitmq-daf-rabbitmq-svc')
# could also use DNS name: 'message-queue'|'message-queue.default.svc.cluster.local'
# for our default

broker_url = 'amqp://%s:%s@%s/%s' % (rabbitmq_user,
                                     rabbitmq_password,
                                     broker_service_host,
                                     rabbitmq_vhost)
# this sets this module up as a celery worker app
app = Celery('tasks', broker=broker_url, backend='amqp')

# this sets up the pike client to do a broadcast of the remeshing event
# which may or may not have a consumer queue
pika_credential = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
parameters = pika.ConnectionParameters(broker_service_host,
                                       5672,
                                       rabbitmq_vhost,
                                       credentials=credentials)
channel = connection.channel()
channel.exchange_declare(exchange=remesh_exchange,
                         exchange_type='fanout')

# this configures this function as a celery task
@app.task
def mesh_lvl2_previews_task(serialized_cg_info, lvl2_node_id,
                            supervoxel_ids,
                            cv_path=None,
                            cv_mesh_dir=None, mip=2, simplification_factor=999999,
                            max_err=40, parallel_download=8, verbose=True,
                            cache_control="no-cache"):

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)
    mesh_lvl2_preview(cg, lvl2_node_id, supervoxel_ids=supervoxel_ids,
                      cv_path=cv_path, cv_mesh_dir=cv_mesh_dir, mip=mip,
                      simplification_factor=simplification_factor,
                      max_err=max_err, parallel_download=parallel_download,
                      verbose=verbose, cache_control=cache_control)

    # now that this meshing worker has completed its task successfully
    # we are broadcasting a message saying what it did
    # compartment classification can consume this message to selectively rerun
    message = {
        'serialized_cg_info': serialized_cg_info,
        'cv_path': cv_path,
        'cv_mesh_dir': cv_mesh_dir,
        'mip': mip,
        'max_err': max_err,
        'parallel_download': parallel_download,
        'supervoxel_ids': supervoxel_ids,
        'lvl2_node_id': lvl2_node_id
    }
    # publish the message to the remesh exchange
    channel.basic_publish(exchange=remesh_exchange,
                          routing_key='',
                          body=json.dumps(message))
