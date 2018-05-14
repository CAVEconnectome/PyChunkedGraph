import numpy as np
import random
import time
import json
import requests
import threading

from google.cloud import pubsub_v1

class NeuromniSimulator():
  """Simulate multiple clients making simultaneous requests of the master.

  This class sets up multiple clients that will execute reads and writes to
  simulate a load on the master and graph server. Each client will record
  when it issues a particular request. We will resever one separate client
  that will record when it receives an update from the master server when any 
  request has been executed.

  Attributes:
    num_clients: number of clients issuing requests to the master server.
    num_client_readers_only: number of clients that will only issue read
      requests to the master server. If 
      num_clients == num_client_readers_only, this would be a low-load
      scenario.
    read_frequency: time in seconds between read requests for each client.
    write_frequency: time in seconds between write requests for each client.
    fraction_write_splits: the fraction of write requests issued that will
      be split requests, as opposed to merge requests. The two requests
      are handled differently by the graph server.
  """
  
  def __init__(self, num_clients=1, num_client_readers_only=1, 
            read_frequency=1, write_frequency=10, 
            fraction_write_splits=0.5):
    self.num_clients = num_clients
    self.num_client_readers_only = num_client_readers_only
    self.read_frequency = read_frequency
    self.write_frequency = write_frequency
    self.fraction_write_splits = fraction_write_splits
    self.clients = []
    self.receiver = None

  def init_clients(self):
    for i in range(self.num_clients):
      root = 432345564227567621
      bbox = (0,0,0,10,10,10)
      self.clients.append(Client(i, root, bbox))
    self.receiver = Receiver('neuromancer-seung-import', 'pychunkedgraph')

  def run(self):
    self.receiver.start()
    for c in self.clients:
      c.start()
    self.receiver.join()
  
  def get_logs(self):
    logs = []
    for c in self.clients:
      logs.extend(c.log)
    return logs


class Client(threading.Thread):
  """Single client simulator.
  """

  def __init__(self, id, bbox, rootA, rootB, 
          read_frequency=1, write_frequency=2, 
                fraction_splits=0.5, runtime=10):
    threading.Thread.__init__(self)
    self.id = id
    self.read_frequency = read_frequency
    self.write_frequency = write_frequency
    self.fraction_splits = fraction_splits
    self.runtime = runtime
    self.bbox = [[0,0,0],[10,10,10]]
    self.supervoxels = [] 
    self.subgraphs = {}
    self.edge = None
    self.log = []

  def read(self, supervoxel):
    root = self.get_root(supervoxel)
    self.get_subgraph(root)

  def get_root(self, supervoxel):
    op = '{0}/root'.format(supervoxel)
    data = {}
    response = self.request(op, data)
    return response['id']

  def get_subgraph(self, root):
    op = 'subgraph'
    data = {"root_id": root, "bbox": self.bbox}
    response = self.request(op, data)
    self.subgraphs[root] = np.array(response['edges'])

  def split(self, edge):
    op = 'split'
    data = {'edge': edge}
    response = self.request(op, data)

  def merge(self, edge, root1, root2):
    op = 'merge'
    data = {'edge': edge}
    response = self.request(op, data)

  def request(self, op, data):
    print((op, data))
    url = 'https://35.231.236.20:4000/1.0/graph/{0}/'.format(op)
    data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    request_time = time.time()
    response = requests.post(url, verify=False, data=data, headers=headers)
    response_time = time.time()
    self.update_log(op, data, response, request_time, response_time)
    return response.json()

  def update_log(self, op, data, response, request_time, response_time):
    master_start = r['time_server_start']
    graph_start = r['time_graph_start']
    graph_stop = r['time_graph_end']    
    entry = [self.id, data, request_time, master_start, graph_start, graph_stop, response_time]
    self.log.append(entry)

  def select_edge(self):
    v1 = np.random.choice(self.subgraphs.values[0].flatten()) 
    v1 = np.random.choice(self.subgraphs.values[1].flatten())
    self.edge = np.array(v1, v2)

  def run(self):
    # start simulation runtime    
    start_time = time.time()
    merge_flag = True
    while time.time() - start_time < self.runtime:
      # always read in the graphs for the same two supervoxels
      # we'll merge then split the same edge from here on out
      if self.read_frequency > 0:
        time.sleep(self.read_frequency)
        supervoxel1 = np.random.choice(self.supervoxels)
        supervoxel2 = np.random.choice(self.supervoxels)
        self.read(supervoxel1)
        self.read(supervoxel2)
        if len(self.subgraphs.keys()) == 1:
          merge_flag = False
      if self.write_frequency > 0:       
        time.sleep(self.write_frequency)
        if merge_flag:         
          self.select_edge()
          self.merge(self.edge)
        else:
          self.split(self.edge)
          self.subgraphs = {}
        merge_flag = !merge_flag

class Receiver(threading.Thread):
  """Simulator of a no-op client that's listening to the pub/sub channel
  """

  def __init__(self, project, subscription_name, runtime=20):
    threading.Thread.__init__(self)
    self.project = project
    self.subscription_name = subscription_name
    self.log = []
    self.runtime = runtime

  def update_log(self, message):
    self.log.append(message)

  def run(self, runtime=60):
      """Receives messages from a pull subscription."""
      subscriber = pubsub_v1.SubscriberClient()
      subscription_path = subscriber.subscription_path(
          self.project, self.subscription_name)

      def callback(message):
          print('Received message: {}'.format(message))
          self.update_log(message)
          message.ack()

      subscriber.subscribe(subscription_path, callback=callback)

      # The subscriber is non-blocking, so we must keep the main thread from
      # exiting to allow it to process messages in the background.
      print('Listening for messages on {}'.format(subscription_path))
      start_time = time.time()
      while time.time() - start_time < self.runtime:
        time.sleep(2)
