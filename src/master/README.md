# Master

The master performs the following functions:
* It is a HTTP server and receives get_root(), get_subgraph(), add_edge() and remove_edge() requests from clients
* It processes requests as they are received and does so on multiple threads, unless the request affects a locked root ID, in which case the request is retried again after 10s for a maximum of 10 tries
* The master handles locking, so that write requests to the same root ID are not processed simultaneously. A lock is applied to a root ID immediately before an edit to the graph is made, and is removed when the edit is complete 
* The master is a publisher and notifies, via a pub-sub system, subscribed clients when a change is made to one of their subscribed root IDs

## Syntax:

### Get Root:
```
https://35.231.236.20:4000/1.0/segment/537753696/root/?
```
or as a cURL request:
```
curl -i http://35.231.236.20:4000/1.0/segment/537753696/root/
```
This returns JSON containing the root ID (if a valid supervoxel ID is presented), along with the time at which the server received the request, the time at which the request started processing the request on the graph, and the time when the request finished processing.
```
{
  "id": "432345564306594611", 
  "time_graph_end": "2018-05-11 01:25:23", 
  "time_graph_start": "2018-05-11 01:25:22", 
  "time_server_start": "2018-05-11 01:25:22"
}
```

### Get Subgraph:
This is a HTTPS POST command, so additional data must be submitted with the request. Until this point, I have tested my server with cURL:
```
curl -X POST -H "Content-Type: application/json" -d '{"root_id":"432345564227567621","bbox":"0, 0, 0, 10, 10, 10"}' http://35.231.236.20:4000/1.0/graph/subgraph/
```
This returns a json file, with entries "edges", "affinities", "time_graph_end", "time_graph_start" and "time_server_start". Example output (warning: not a complete subgraph; subgraphs are normally much larger):
```
{
  "affinities": [
    0.6716280579566956, 
    0.6449402570724487, 
    0.8567864298820496, 
    0.8496066331863403, 
    0.8754334449768066],
 "edges": [
    [
      73184602046335106, 
      73184602046335107
    ], 
    [
      73184602046335106, 
      73184602046335148
    ], 
    [
      73184602046335106, 
      73184602046335149
    ], 
    [
      73184602046335106, 
      73184602046334951
    ], 
    [
      73184602046335106, 
      73184602046334986
    ]
  ],
  "time_graph_end": "2018-05-11 01:49:31", 
  "time_graph_start": "2018-05-11 01:49:13", 
  "time_server_start": "2018-05-11 01:49:13"
}
``` 

### Add Edge:
This is another HTTP POST command:
```
curl -X POST -H "Content-Type: application/json" -d '{"edge":"537753696, 537544567"}' http://35.231.236.20:4000/1.0/graph/merge/
```
This returns the new root ID for the merged supervoxels as a string in a JSON file, or returns "NaN" if either the edge cannot be added due to such an edge already existing, or if at least one of the root IDs for the implicated supervoxels is locked for the whole duration of the request (up to 100 seconds - the master attempts to fulfil the merge request 10 times, with each try separated by 10 seconds, but gives up after that).  In this latter case, the user may want to resend the request. The client also receives back the times (in UTC) recorded by the master when the initial HTTP request was received ("time_server_start"), the time at which the edit to the graph started ("time_graph_start"), and the time when the edit to the graph concluded ("time_graph_end"). If the edit to the graph was unsuccessful, this latter field will be "NaN"; if the root ID for at least one of the implicated supervoxels was locked for the full duration of the request, and no edit to the graph ever started, both the "time_graph_start" and "time_graph_end" fields will be "NaN". Example response:
```
{
  "new_root_id": "NaN",
  "time_graph_end": "NaN",
  "time_graph_start": "2018-05-11 01:07:59"
  "time_server_start": "2018-05-11 01:07:58"
		
}

```

### Remove Edge:

```
curl -X POST -H "Content-Type: application/json" -d  '{"edge":"537753696, 537544567"}' http://35.231.236.20:4000/1.0/graph/split/
```
This returns the new root IDs for the supervoxels affected by the edge removal (as a string with the two root IDs separated by comma and space), or returns "None" if the edge cannot be split due to it not existing in the first place, or "NaN" if one of the root IDs for the implicated supervoxels is locked for the whole duration of the request (up to 100 seconds - the master attempts to fulfil the split request 10 times, with each try separated by 10 seconds, but gives up after that).  In this latter case, the user may want to resend the request.  The client also receives back the times (in UTC) recorded by the master when the initial HTTP request was received ("time_server_start"), the time at which the edit to the graph started ("time_graph_start"), and the time when the edit to the graph concluded ("time_graph_end"). If the edit to the graph was unsuccessful, this latter field will be "NaN"; if the root ID for at least one of the implicated supervoxels was locked for the full duration of the request, and no edit to the graph ever started, both the "time_graph_start" and "time_graph_end" fields will be "NaN". Example response:

```
{
  "new_root_ids": "None", 
  "time_graph_end": "2018-05-11 01:18:16", 
  "time_graph_start": "2018-05-11 01:18:16", 
  "time_server_start": "2018-05-11 01:18:15"
}

```


