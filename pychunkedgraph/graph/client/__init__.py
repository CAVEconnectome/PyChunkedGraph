"""
Sub packages/modules for backend storage clients
Currently supports Google Big Table

A simple client needs to be able to create the graph,
store graph meta and to write and read node information.
Also needs locking support to prevent race conditions
when modifying root/parent nodes.

In addition, clients with more features like generating unique IDs
and logging facilities can be implemented by inherting respective base classes.

These methods are in separate classes because they are logically related.
This also makes it possible to have different backend storage solutions,
making it possible to use any unique features these solutions may provide.

Please see `base.py` for more details.
"""
