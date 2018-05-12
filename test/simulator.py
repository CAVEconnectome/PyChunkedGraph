from urllib import request
import random
import time

class MultiClientSimulator():
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
		self.num_client_readers = num_client_readers
		self.read_frequency = read_frequency
		self.write_frequency = write_frequency
		self.fraction_write_splits = fraction_write_splits
		self.clients = []

class Client():
	"""Single client simulator.
	"""

	def __init__(self, id, read_frequency=1, write_frequency=10, 
											fraction_write_splits=0.5):
		self.id = id
		self.read_frequency = read_frequency
		self.write_frequency = write_frequency
		self.fraction_write_splits = fraction_write_splits
		self.root = 10
		self.bbox = (0,0,0,10,10,10)
		self.subgraph = None
		self.log = []


	def read(self, root, bbox):
		op = 'subgraph'
		url = 'https://35.231.236.20:4000/1.0/graph/{0}/?'.format(op)
		data = '{{"root_id":"{0}","bbox":"{1}"}}'.format(root, bbox)
		self.update_log(url + data)
		# request.urlopen(url, data=data)
		print(url+data)
		return 

	def write(self, edge):
		op = 'merge'
		if random() < self.fraction_write_splits:
			op = 'split'
		url = 'https://35.231.236.20:4000/1.0/graph/{0}/?'.format(op)
		data = '{{"edges":[["{0}, {1}"]]}}'.format(edge[0], edge[1])
		self.update_log(url + data)
		# request.urlopen(url, data=data)
		print(url+data)

	def update_log(self, request):
		entry = [self.id, request, time.time()]
		self.log.append(entry)

	def pick_edge(self, subgraph):
		# return two random nodes from subgraph
		return (5,6)

	def exec(self, runtime=20):
		start_time = time.time()
		while time.time() - start_time < runtime:
			if self.read_frequency > 0:
				time.sleep(self.read_frequency)
				self.subgraph = self.read(self.root, self.bbox)
			if self.write_frequency > 0:
				time.sleep(self.write_frequency)
				edge = self.pick_edge(self.subgraph)
				self.write(edge)





