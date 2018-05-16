# Simulator  
The simulator is a tool to stress test the Neuromni system. It simulates a 
group of clients that issue read and write requests to the Master server, and 
includes listen-only clients called Receivers, the poll for updates over 
the Neuromni pub/sub channel. 

Initial client request times are recorded, the 
Master server includes processing times in its response, and the Receiver logs 
when it records an update.

## Requirements
Implemented with python3.

```
pip install -r requirements.txt
```

## Quickstart
```
import simulator as s

# create receiver & start pulling for messages from the pub/sub
r = s.Receiver('neuromancer-seung-import', 'MySub')
r.start()

# create client #5, selecting from set of supervoxels
# load subgraphs for two of the supervoxels
# find an edge between the two subgraphs
# merge the two subgraphs based on that edge, then split it
c = s.Client(5, [50826, 57496]) 
c.load_subgraphs()
c.select_edge_to_merge()
c.merge()
c.split()

# inspect the logs of both the client & receiver
c_df = c.export_log()
r_df = r.export_log()
```

To run the simulator under the conditions in our [cos518 post](https://medium.com/@sven.dorkenwald/building-a-multi-user-platform-for-real-time-very-large-graph-editing-ee37268025ad), 
run the `project_simulations()` method.
