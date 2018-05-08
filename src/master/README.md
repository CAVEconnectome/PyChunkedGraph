# Master
The master performs the following functions:
* It is a HTTPS server and receives get_root(), get_subgraph(), add_edge(), remove_edge() requests from clients
* It processes get_root() reads immediately; and queues the other requests for processing on multiple threads (a natural extension would be to enable the processing of these requests on multiple cores)
* The master is a publisher and notifies, via a pub-sub system, subscribed clients when a change is made to one of their subscribed root IDs

## Syntax:

### Get Root:
```
https://35.231.236.20:4000/1.0/segment/537753696/root/?
```
returns the root ID in the returned HTTPS header.

### Get Subgraph:
This is a HTTPS POST command, so additional data must be submitted with the request. Until this point, I have tested my server with cURL:
```
curl -d '{"root_id":"432345564227567621","bbox":"0, 0, 0, 10, 10, 10"}' --insecure -i https://35.231.236.20:4000/1.0/graph/subgraph/?
```

### Add Edge:
This is another HTTPS POST command:
```
curl -d '{"edges":"537753696, 537544567"}' --insecure -i https://35.231.236.20:4000/1.0/graph/merge/?
```

### Remove Edge:

```
curl -d '{"edges":"537753696, 537544567"}' --insecure -i https://localhost:4000/1.0/graph/split/?
```