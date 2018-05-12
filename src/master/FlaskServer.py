from flask import Flask, jsonify, Response, request
from threading import Lock
from flask_cors import CORS
import sys
#sys.path.insert(0, '/home/zashwood/PyChunkedGraph/src/pychunkedgraph') 
sys.path.insert(0, '/code/src/pychunkedgraph') #Include Sven's pychunkedgraph code
#sys.path.insert(0, '/usr/people/zashwood/Documents/PyChunkedGraph/src/pychunkedgraph') #Include Sven's pychunkedgraph code
import chunkedgraph #Import chunkedgraph script 
import numpy as np
import time
from time import gmtime, strftime
import redis
from google.cloud import pubsub_v1
# curl -i https://localhost:4000/1.0/segment/537753696/root/
# SPLIT:
  #  curl -X POST -H "Content-Type: application/json" -d  '{"edge":"537753696, 537544567"}' --insecure -i https://localhost:4000/1.0/graph/split/
# MERGE:
    #curl -X POST -H "Content-Type: application/json" -d '{"edge":"537753696, 537544567"}' --insecure -i https://localhost:4000/1.0/graph/merge/
# GET SUBGRAPH 
   # curl -X POST -H "Content-Type: application/json" -d '{"root_id":"432345564227567621","bbox":"0, 0, 0, 10, 10, 10"}' --insecure -i https://localhost:4000/1.0/graph/subgraph/ >> subgraph.txt

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
   host="redis", port=6379, charset="utf-8", decode_responses=True)

@app.route('/')
def index():
    return "PyChunkedGraph Server"

@app.route('/1.0/segment/<atomic_id>/root/', methods=['GET'])
def handle_root(atomic_id):
    time_server_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    try:
    	time_graph_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    	root_id = cg.get_root(int(atomic_id))
    	time_graph_end = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    	return jsonify({"id": str(root_id), "time_server_start":time_server_start, "time_graph_start":time_graph_start, "time_graph_end":time_graph_end})
    except Exception as e:
    	print(e)
    	time_graph_end = 'NaN'
    	return jsonify({"id": 'NaN', "time_server_start":time_server_start, "time_graph_start":time_graph_start, "time_graph_end":time_graph_end})

@app.route('/1.0/graph/merge/', methods=['POST'])
def handle_merge():
	# Record time when request received by master
    time_server_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
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
                # Numpy arrays of historical root IDs 
                historical1 = cg.read_agglomeration_id_history(root1)
                historical2 = cg.read_agglomeration_id_history(root2)
                all_historical_ids = np.append(historical1, historical2)
                print(all_historical_ids)
                # Now check if any of the historical IDs are being processed in redis DB: (inefficient - will not scale to large #s of historical IDs)
                for i, historic_id in enumerate(all_historical_ids):
                    status = redis_conn.get(str(historic_id))
                    print(status)
                    if status == 'busy':
                        #print(status)
                        # Come out of loop and wait sleep_time seconds before rechecking all_historical_ids
                        raise Exception('locked_id')
                # Now try to perform write (providing exception has not been raised)
                time_graph_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                try:
                    # If have made it through without raising exception, IDs are not locked and add_edge can be performed safely
                    # First add root1 and root2 to locked list:
                    redis_conn.set(str(root1), "busy")
                    redis_conn.set(str(root2), "busy")
                    # Perform edit on graph
                    out = cg.add_edge(edge)
                    time_graph_end = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    # Now remove root1 and root2 from redis (unlock these root IDs)
                    redis_conn.delete(str(root1))
                    redis_conn.delete(str(root2))
                except Exception as e: # Raises exception if problem with add_edge function and provided IDs
                    time_graph_end = 'NaN'
                    print(e)
                    # Now remove root1 and root2 from redis (unlock these root IDs) even if exception raised
                    redis_conn.delete(str(root1))
                    redis_conn.delete(str(root2))
                    out = 'NaN'
                    return jsonify({"new_root_id": str(out), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})
                else: # if write to graph was successful:
                     # Publish edit to pubsub system
                    msg = u'merge {}'.format(edge)
                    # Encode merge message as byte string 
                    msg = msg.encode('utf-8')
                    # Root IDs as byte string:
                    msg_old_roots = str(root1) + ', ' + str(root2)
                    msg_new_root = str(out)
                    publisher.publish(topic_path, msg, old_root_ids = msg_old_roots, new_root_ids = msg_new_root)
                    # Return new root as JSON
                    return jsonify({"new_root_id": str(out), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})  
            except Exception as inst:
                print(inst.args)
                time_graph_start = 'NaN'
                time_graph_end = 'NaN'
                attempts += 1
                time.sleep(sleep_time)
        # If make it to this point, while loop has failed and user should try again
        out = 'NaN'
        return jsonify({"new_root_id": str(out), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})  
    else: #Client has not entered valid supervoxel IDs
        return '', 400

        
@app.route('/1.0/graph/split/', methods=['POST'])
def handle_split():
    # Record time when request received by master
    time_server_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
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
                # Get root IDs for both supervoxels: (should be same for split)
                root1 = cg.get_root(int(edge[0]))
                root2 = cg.get_root(int(edge[1]))
                # Numpy arrays of historical root IDs 
                historical1 = cg.read_agglomeration_id_history(root1)
                historical2 = cg.read_agglomeration_id_history(root2)
                all_historical_ids = np.append(historical1, historical2)
                print(all_historical_ids)
                # Now check if any of the historical IDs are being processed in redis DB: (inefficient - will not scale to large #s of historical IDs)
                for i, historic_id in enumerate(all_historical_ids):
                    status = redis_conn.get(str(historic_id))
                    print(status)
                    if status == 'busy':
                        #print(status)
                        # Come out of loop and wait sleep_time seconds before rechecking all_historical_ids
                        raise Exception('locked_id')
                # Now try to perform write (providing exception has not been raised)
                time_graph_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                try:
                    # If have made it through without raising exception, IDs are not locked and add_edge can be performed safely
                    # First add root1 and root2 to locked list:
                    redis_conn.set(str(root1), "busy")
                    redis_conn.set(str(root2), "busy")
                    out = cg.remove_edge(edge)
                    time_graph_end = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    # Now remove root1 and root2 from redis (unlock these root IDs)
                    redis_conn.delete(str(root1))
                    redis_conn.delete(str(root2))
                except Exception as e: # Raises exception if problem with add_edge function and provided IDs
                    time_graph_end = 'NaN'
                    print(e)
                    # Now remove root1 and root2 from redis (unlock these root IDs) even if exception raised
                    redis_conn.delete(str(root1))
                    redis_conn.delete(str(root2))
                    out = 'NaN'
                    return jsonify({"new_root_ids": str(out), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})
                else: # if edit to graph was successful, publish edit to pubsub system and return JSON containing new Root IDs
                     # Publish edit to pubsub system
                    msg = u'split {}'.format(edge)
                    # Encode merge message as byte string 
                    msg = msg.encode('utf-8')
                    # Root IDs as byte string:
                    msg_old_root = str(root1) 
                    msg_new_root = str(out)
                    publisher.publish(topic_path, msg, old_root_id = msg_old_root, new_root_ids = msg_new_root)
                    return jsonify({"new_root_ids": str(out), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})               
            except Exception as inst:
                print(inst.args)
                time_graph_start = 'NaN'
                time_graph_end = 'NaN'
                attempts += 1
                time.sleep(sleep_time)
        # If make it to this point, while loop has failed and user should try again     
        out = 'NaN'
        return jsonify({"new_root_ids": str(out), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})  
    else: #Client has not entered valid supervoxel IDs
        return '', 400


@app.route('/1.0/graph/subgraph/', methods=['POST'])
def get_subgraph():
# Record time when request received by master
    time_server_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # Collect edges from json:
    if 'root_id' in request.get_json() and 'bbox' in request.get_json():
        time_graph_start = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        try:
            root_id = int(request.get_json()['root_id'])
            bounding_box  = np.reshape(np.array(request.get_json()['bbox'].split(','), dtype = int), (2,3))
            edges, affinities = cg.get_subgraph(root_id, bounding_box)
            time_graph_end = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            return jsonify({"edges":edges.tolist(), 'affinities':affinities.tolist(), "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})   
        except Exception as e:
            print(e)
            edges, affinities = 'NaN', 'NaN'
            time_graph_end = 'NaN'
            return jsonify({"edges":edges, 'affinities':affinities, "time_server_start": time_server_start, "time_graph_start": time_graph_start, "time_graph_end":time_graph_end})
    else: # Case where client has supplied inappropriate root_id/bbox
        return '', 400


if __name__ == '__main__':
    # Initialize chunkedgraph:
    cg = chunkedgraph.ChunkedGraph(dev_mode=False)
    # Initialize google pubsub publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('neuromancer-seung-import', 'pychunkedgraph')
    app.run(host = '0.0.0.0', port = 4000, debug = True, threaded=True, ssl_context = ('keys/server.crt', 'keys/server.key'))
