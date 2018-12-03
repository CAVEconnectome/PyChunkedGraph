from celery import Celery
from celery.utils.log import get_task_logger
import pika
import os

logger = get_task_logger(__name__)

# set username and password for broker, with overrides from environment variables
rabbitmq_user = os.environ.get('RABBITMQ_USERNAME', 'rabbit')
rabbitmq_password = os.environ.get('RABBITMQ_PASSWORD', 'rabbitmq')
rabbitmq_vhost = os.environ.get('RABBITMQ_VHOST', '/')
remesh_exchange = os.environ.get('RABBIGMQ_REMESH_EXCHANGE', 'remeshing')
# Get Kubernetes-provided address of the broker service
broker_service_host = os.environ.get(
    'MESSAGE_QUEUE_SERVICE_HOST', 'localhost')
# could also use DNS name: 'message-queue'|'message-queue.default.svc.cluster.local'
# for our default

broker_url = 'amqp://%s:%s@%s/%s' % (rabbitmq_user,
                                     rabbitmq_password,
                                     broker_service_host,
                                     rabbitmq_vhost)
# this sets this module up as a celery worker app
app = Celery('tasks', broker=broker_url, backend='rpc')

# this sets up the pika client to do a broadcast of the remeshing event
# which may or may not have a consumer queue
pika_credential = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
parameters = pika.ConnectionParameters(broker_service_host,
                                       virtual_host=rabbitmq_vhost,
                                       credentials=pika_credential)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.exchange_declare(exchange=remesh_exchange,
                         exchange_type='fanout')