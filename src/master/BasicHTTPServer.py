#!/usr/bin/env python
"""
Basic server code originally from https://gist.github.com/bradmontgomery/2219997
Modified to incorporate features in https://github.com/seung-lab/chunked-graph/blob/master/src/server.jl
by Zoe Ashwood (April 2018)

Usage::
    ./dummy-web-server.py [<port>]
Send a GET request::
    curl http://localhost
Send a HEAD request::
    curl -I http://localhost
Send a POST request::
    curl -d "foo=bar&bin=baz" http://localhost
"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import json #To parse JSON system configuration file
import ssl #For SSL encryption
import os #To make directories as appropriate and to execute terminal commands


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()
        
    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers()
        self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        
def run(server_class=HTTPServer, handler_class=S, port=9100, certfile='./server.pem'):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    #TODO: implement SSL wrapping
    httpd.socket = ssl.wrap_socket (httpd.socket, certfile = certfile, server_side=True)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    # Read in JSON file detailing server configuration
    settings = json.load(open('server_conf.json'))
    # Establish appropriate directories 
    os.system(('mkdir -p ' + settings['graphpath']))
    os.system(('mkdir -p ' + settings['logpath']))
    os.system(('mkdir -p ' + settings['certpath']))

    # TODO: obtain SSL certificate and key; if don't already exist, create these

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()