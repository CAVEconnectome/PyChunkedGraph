#!/usr/bin/env python
"""
Basic server code originally from https://gist.github.com/bradmontgomery/2219997
Modified to incorporate features in https://github.com/seung-lab/chunked-graph/blob/master/src/server.jl
by Zoe Ashwood (April 2018)

## HTTP server accepts requests from client and writes these to a DB for later ordering
## and processing by TaskManager.py

Usage::
    ./BasicHTTPServer.py [<port>]
Send a GET request via the terminal (assuming server running locally on port 9100)::
    curl --insecure -i https://localhost:9100/1.0/segment/1000/root/?   
Send a HEAD request via the terminal (assuming server running locally on port 9100)::
    curl -I --insecure https://localhost:9100
Send a POST request (assuming server running locally on port 9100)::
    curl -d "foo=bar&bin=baz" --insecure -i https://localhost:9100/1.0/graph/split/?
"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import json #To parse JSON system configuration file
import ssl #For SSL encryption
import os #To make directories as appropriate and to execute terminal commands
from pathlib2 import Path #To check if certificate exists
import re #For regular expressions

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    # Returns information describing server: e.g. server, content type, data, request version (e.g. HTTP/1.0)      
    def do_HEAD(self):
        self._set_headers()

    # Handle HTTP GET requests  
    def do_GET(self):
    	# handle each type of HTTP GET request. NB self.path is request
    	# Get root request:
    	if(re.match(r"/1.0/segment/(\d+)/root/?", self.path)):
    		print "Get root request received" #Return root ID as body of HTTP response
        	self._set_headers()
        	self.send_response(200, "Test response")
        	#self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Get active operations (items in write queue)
        elif(re.match(r"/1.0/graph/active/?", self.path)):	
        	print "Get active operations request received"
        	# Return list of active write operations
        	self._set_headers()
        	self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        else:
        	print "could not parse " + self.path
        	self._set_headers()
        	# TODO: return error message

     
    # Handle merge and split operations by appending these to an outside DB    
    def do_POST(self):
        # handle each type of HTTP POST request. NB self.path is request
    	# Merge supervoxels:
    	if(re.match(r"/1.0/graph/merge/?", self.path)):
    		print "Merge request received" #Return root ID as body of HTTP response
        	self._set_headers()
        	self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Handle split
        elif(re.match(r"/1.0/graph/split/?", self.path)):
    		print "Split request received" #Return root ID as body of HTTP response
        	self._set_headers()
        	self.wfile.write("<html><body><h1>hi!</h1></body></html>")	
        # Subscribe edges request:	
        elif(re.match(r"/1.0/segment/(\d+)/subscribe/?", self.path)):
        	print "Subscribe edges request received"
        	self._set_headers()
        	self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Unsubscribe edges request:		
        elif(re.match(r"/1.0/segment/(\d+)/unsubscribe/?", self.path)):	
        	print "Unsubscribe edges request received"
        	self._set_headers()
        	self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        # Save graph	
        elif(re.match(r"/1.0/graph/save/?", self.path)):	
        	print "Save request received"
        	self._set_headers()
        	self.wfile.write("<html><body><h1>hi!</h1></body></html>")
        else:
        	print "could not parse " + self.path
        	self._set_headers()
        	#TODO: return error message			
        
def run(certfile, server_class=HTTPServer, handler_class=S, port=9100):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    #Wrap socket in SSL context
    httpd.socket = ssl.wrap_socket (httpd.socket, certfile = certfile, server_side=True)
    print 'Starting httpd...'
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

    if len(argv) == 2:
        run(certfile, port=int(argv[1]))
    else:
        run(certfile)


