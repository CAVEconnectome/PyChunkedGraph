#!/usr/bin/env python
"""
Basic server code originally from https://gist.github.com/bradmontgomery/2219997
Modified to incorporate features in https://github.com/seung-lab/chunked-graph/blob/master/src/server.jl
by Zoe Ashwood (April 2018)

## HTTP server accepts requests from client and writes these to a DB for later ordering
## and processing by TaskManager.py

Usage::
    ./BasicHTTPServer.py [<port>]
Send a GET request::
    curl --insecure https://localhost:9100
Send a HEAD request::
    curl -I --insecure https://localhost:9100
Send a POST request::
    curl -d --insecure "foo=bar&bin=baz" https://localhost:9100
"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import json #To parse JSON system configuration file
import ssl #For SSL encryption
import os #To make directories as appropriate and to execute terminal commands
from pathlib2 import Path #To check if certificate exists

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    # Serve read requests immediately with server    
    def do_GET(self):
        self._set_headers()
        self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()
     
    # Handle merge and split operations by appending these to an outside DB    
    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers()
        self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        
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


