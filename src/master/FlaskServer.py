from flask import Flask, jsonify, Response, request
from threading import Lock
from flask_cors import CORS
import sys
#sys.path.insert(0, '/home/zashwood/PyChunkedGraph/src/pychunkedgraph') #Include Sven's pychunkedgraph code
sys.path.insert(0, '/usr/people/zashwood/Documents/PyChunkedGraph/src/pychunkedgraph') #Include Sven's pychunkedgraph code
import chunkedgraph #Import chunkedgraph script 
import numpy as np
import time
import redis
# curl -i http://localhost:4000/1.0/segment/537753696/root/
# SPLIT:
  #  curl -X POST -H "Content-Type: application/json" -d  '{"edge":"537753696, 537544567"}' http://localhost:4000/1.0/graph/split/
# MERGE:
    #curl -X POST -H "Content-Type: application/json" -d '{"edge":"537753696, 537544567"}' http://localhost:4000/1.0/graph/merge/
# GET SUBGRAPH 
   # curl -X POST -H "Content-Type: application/json" -d '{"root_id":"432345564227567621","bbox":"0, 0, 0, 10, 10, 10"}' http://localhost:4000/1.0/graph/subgraph/

app = Flask(__name__)
CORS(app)

# Global variables:
# Maximum number of allowed tries for add edge/remove edge command before giving up (either command itself cannot be implemented or
# a root id is locked for all tries)
max_tries = 10
# Time in seconds before retry (add edge/remove edge)
sleep_time = 10

#Redis host
redis_conn = redis.StrictRedis(
   host="localhost", port=6379, charset="utf-8", decode_responses=True)

@app.route('/')
def index():
    return ""

@app.route('/1.0/segment/<atomic_id>/root/', methods=['GET'])
def handle_root(atomic_id):
	#Read and write to redis
    #redis_conn.get("/id/%s"%str(atomic_id),"busy")
    #redis_conn.set("/id/%s"%str(atomic_id),"busy")

    root_id = cg.get_root(int(atomic_id))
    print(root_id)
    return jsonify({"id": str(root_id)})

@app.route('/1.0/graph/merge/', methods=['POST'])
def handle_merge():
    # Collect edges from json:
    if 'edge' in request.get_json():
        edge = request.get_json()['edge']
    # Obtain edges from request dictionary, and convert to numpy array with uint64s
        edge = np.fromstring(edge, sep = ',', dtype = np.uint64)
        # Now try, for a maximum of max_tries, to add edge
        attempts = 0 
        while attempts < max_tries:
        	try:
        		#Check if either of the root_IDs are being processed by any of the threads currently: 
        		# Get root IDs for both supervoxels:
        		root1 = cg.get_root(int(edge[0]))
        		root2 = cg.get_root(int(edge[1]))
        	
        		# TODO: update these when cg.read_agglomeration_id_history() is working again
        		historical1 = cg.read_agglomeration_id_history(root1)
        		historical2 = cg.read_agglomeration_id_history(root2)
        		#historical1 = [root1]
        		#historical2 = [root2]
        		all_historical_ids = np.append(historical1, historical2)
        		print(all_historical_ids)
        		# Now check if any of the historical IDs are being processed in redis DB: (inefficient - will not scale to large #s of historical IDs)
        		for i, historic_id in enumerate(all_historical_ids):
        			status = redis_conn.get(str(historic_id))
        			if status == 'busy':
        				print(status)
        				# Come out of loop and wait sleep_time seconds before rechecking all_historical_ids
        				raise Exception('locked_id')
        		# If have made it through without raising exception, IDs are not locked and add_edge can be performed safely
        		# First add root1 and root2 to locked list:
        		redis_conn.set(str(root1), "busy")
        		redis_conn.set(str(root2), "busy")
        		# Now try to perform write
        		try:
        			out = cg.add_edge(edge)
        			# Now remove root1 and root2 from redis (unlock these root IDs)
        			redis_conn.delete(str(root1))
        			redis_conn.delete(str(root2))
        			return jsonify({"new_root_id": str(out)})
        		except Exception as e: # Raises exception if problem with add_edge function and provided IDs
        			print(e)
        			# Now remove root1 and root2 from redis (unlock these root IDs) even if exception raised
        			redis_conn.delete(str(root1))
        			redis_conn.delete(str(root2))
        			out = 'NaN'
        			return jsonify({"new_root_id": str(out)})        		
        	except Exception as inst:
        		attempts += 1
        		time.sleep(sleep_time)
        # If make it to this point, while loop has failed and user should try again		
        out = 'NaN'
        return jsonify({"new_root_id": str(out)})
    else:
    	return '', 400

        
@app.route('/1.0/graph/split/', methods=['POST'])
def handle_split():
    #Read and write to redis
    #redis_conn.get("/id/%s"%str(atomic_id),"busy")
    #redis_conn.set("/id/%s"%str(atomic_id),"busy")
    # Collect edges from json:
    if 'edge' in request.get_json():
        edge = request.get_json()['edge']
    # Obtain edges from request dictionary, and convert to numpy array with uint64s
        edge = np.fromstring(edge, sep = ',', dtype = np.uint64)
        try: 
        	out = cg.remove_edge(edge)
        except:
        	out = 'NaN'
        return jsonify({"new_root_ids": str(out)})
    else: 
    	return '', 400

@app.route('/1.0/graph/subgraph/', methods=['POST'])
def get_subgraph():
    #Read and write to redis
    #redis_conn.get("/id/%s"%str(atomic_id),"busy")
    #redis_conn.set("/id/%s"%str(atomic_id),"busy")
    # Collect edges from json:
    if 'root_id' in request.get_json() and 'bbox' in request.get_json():
    	root_id = int(request.get_json()['root_id'])
    	bounding_box  = np.reshape(np.array(request.get_json()['bbox'].split(','), dtype = int), (2,3))
    	try:
    		edges, affinities = cg.get_subgraph(root_id, bounding_box)
    	except:
    		edges, affinities = 'NaN', 'NaN'
    	return jsonify({"edges":edges.tolist(), 'affinities':affinities.tolist()})	
    else: 
    	return '', 400


if __name__ == '__main__':
    # Initialize chunkedgraph:
    cg = chunkedgraph.ChunkedGraph(dev_mode=False)
    app.run(host = 'localhost', port = 4000, debug = True, threaded=True)