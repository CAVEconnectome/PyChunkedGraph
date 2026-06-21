"""Lazy graph_tool surface: import this module (not graph_tool) so the heavy
import and its scipy pull are deferred to first use, not package-import time.
"""

import graph_tool
import graph_tool.flow as flow
import graph_tool.topology as topology
from graph_tool import Graph, GraphView
