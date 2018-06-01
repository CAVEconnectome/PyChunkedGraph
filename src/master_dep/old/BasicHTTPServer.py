#!/usr/bin/env python
"""
Basic server code originally from https://gist.github.com/bradmontgomery/2219997
Modified to incorporate features in https://github.com/seung-lab/chunked-graph/blob/master_dep/src/server.jl
by Zoe Ashwood (April 2018)

## HTTP server accepts requests from client and writes these to a DB for later ordering
## and processing by TaskManager.py

Usage::
    ./BasicHTTPServer.py [<port>]
Send a GET request via the terminal (assuming server running locally on port 4000)::
    curl --insecure -i https://localhost:4000/1.0/segment/1000/root/?   
Send a HEAD request via the terminal (assuming server running locally on port 4000)::
    curl -I --insecure https://localhost:4000
Send a POST request (assuming server running locally on port 4000)::
# SPLIT:
    curl -d '{"edge":"537753696"}' --insecure -i https://localhost:4000/1.0/graph/split/?
# MERGE:
    curl -d '{"edges":"537753696, 537544567"}' --insecure -i https://localhost:4000/1.0/graph/merge/?
# GET SUBGRAPH 
    curl -d '{"root_id":"432345564227567621","bbox":"0, 0, 0, 10, 10, 10"}' --insecure -i https://localhost:4000/1.0/graph/subgraph/?
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import json #To parse JSON system configuration file
import ssl #For SSL encryption
import os #To make directories as appropriate and to execute terminal commands
from pathlib2 import Path #To check if certificate exists
import re #For regular expressions
import sys #To add '../pychunkedgraph' directory as source of python code  
#sys.path.insert(0, '/home/zashwood/PyChunkedGraph/src/pychunkedgraph') #Include Sven's pychunkedgraph code
sys.path.insert(0, '/usr/people/zashwood/Documents/PyChunkedGraph/src/pychunkedgraph') #Include Sven's pychunkedgraph code
import chunkedgraph #Import chunkedgraph script 
import numpy as np
import ast #For processing of byte strings and converting them into dictionaries

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    # Returns information describing server: e.g. server, content type, data, request version (e.g. HTTP/1.0)      
    def do_HEAD(self):
        self._set_headers()

    # Handle HTTPS GET requests  
    def do_GET(self):
        # handle each type of HTTP GET request. NB self.path is request
        # Get root request:
        if(re.match(r"/1.0/segment/(\d+)/root/?", self.path)):
            print("Get root request received") #Return root ID as body of HTTP response
            self._set_headers()
            # First extract atomic ID from request:
            match = re.search('/segment/(\d+)', self.path)
            atomic_id = int(match.group(1))
            # Now extract Root ID from chunked graph
            root_id = cg.get_root(atomic_id)
            inner_out_str = 'Root ID = ' + str(root_id)
            self.send_response(200, inner_out_str)
            self.end_headers()
            #out_str = "<html><body><h1>" + inner_out_str + "</h1></body></html>"
            #self.wfile.write(out_str.encode("utf-8"))
            #self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Get active operations (items in write queue)
        #elif(re.match(r"/1.0/graph/active/?", self.path)):
            #print("Get active operations request received")
            # Return list of active write operations
            #self._set_headers()
            #self.end_headers()
            #self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        else:
            print("could not parse " + self.path)
            self._set_headers()
            self.send_response(400)
            self.end_headers()

     
    # Handle merge and split operations by appending these to an outside DB    
    def do_POST(self):
        # handle each type of HTTP POST request. NB self.path is request
        # Merge supervoxels:
        if(re.match(r"/1.0/graph/merge/?", self.path)):
            print("Merge request received")
            request = self.rfile.read(int(self.headers.get('content-length')))
            request = request.decode("utf-8") #Decode byte string object
            request = ast.literal_eval(request) #Convert to dictionary object
            # Obtain edges from request dictionary, and convert to numpy array with uint64s
            edges = np.fromstring(request["edges"], sep = ',', dtype = np.uint64)
            out = cg.add_edge(edges) 
            self._set_headers()
            self.send_response(200, out)
            self.end_headers()
            #self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Handle split
        elif(re.match(r"/1.0/graph/split/?", self.path)):
            print("Split request received") 
            request = self.rfile.read(int(self.headers.get('content-length')))
            request = request.decode("utf-8") #Decode byte string object
            request = ast.literal_eval(request) #Convert to dictionary object
            # Obtain edges from request dictionary, and convert to numpy array with uint64s
            edge = np.fromstring(request["edge"], sep = ',', dtype = np.uint64)
            out = cg.remove_edge(edge)
            self._set_headers()
            self.send_response(200, out)
            self.end_headers()
            #self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Handle get_subgraph
        elif(re.match(r"/1.0/graph/subgraph/?", self.path)):
            print("Get subgraph request received") 
            request = self.rfile.read(int(self.headers.get('content-length')))
            request = request.decode("utf-8") #Decode byte string object
            request = ast.literal_eval(request) #Convert to dictionary object
            root_id = int(request["root_id"])
            bounding_box  = np.reshape(np.array(request["bbox"].split(','), dtype = int), (2,3))
            edges, affinities = cg.get_subgraph(root_id, bounding_box)
            json_out = json.dumps({'edges':edges.tolist(), 'affinities':affinities.tolist()})
            self._set_headers()
            self.send_response(200, json_out)
            self.end_headers()
        else:
            print("could not parse " + self.path)
            self._set_headers()
            self.send_response(400)
            self.end_headers()
            #TODO: return error message
        
def run(certfile, server_class=HTTPServer, handler_class=S, port=4000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    #Wrap socket in SSL context
    httpd.socket = ssl.wrap_socket (httpd.socket, certfile = certfile, server_side=True)
    print('Starting httpd on port ' + str(port) + '...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    # Read in JSON file detailing server configuration
    settings = json.load(open('server_conf.json'))
    # Establish appropriate directories 
    os.system(('mkdir -p ' + settings['certpath']))

    # Certificate and keyfile for SSL encryption
    certfile = settings['certpath']+'/server.pem'

    # Check if certificate and key exist; if not, create these
    if not Path(certfile).is_file():
        create_command = 'openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ' + certfile + ' -out ' + certfile
        os.system(create_command)

    # Initialize chunkedgraph:
    cg = chunkedgraph.ChunkedGraph(dev_mode=False)

    if len(argv) == 2:
        run(certfile, port=int(argv[1]))
    else:
        run(certfile)


