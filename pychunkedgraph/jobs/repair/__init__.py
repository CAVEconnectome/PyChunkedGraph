"""
Repair failed edit operations.
There is small possibility of an edit operation failing
in the middle of persisting changes, this can corrupt the
chunkedgraph. When this happens a root is locked indefinitely
to prevent any more edits on it until the issue is fixed.
This is meant to be run manually after inspecting such cases.
"""
