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