"""Resolve the configured SV splitter implementation.

`PCG_SV_SPLITTER` env var holds the dotted import path of the Splitter
class. To use a different splitter, install its package and set the
env var — no PCG code change required.
"""

import importlib
import os

DEFAULT_SPLITTER = "supervoxel_splitter.GeodesicSplitter"


def get_splitter(**kwargs):
    """Import the configured Splitter class and instantiate.

    `**kwargs` forward to the class constructor so caller-side tuning
    propagates without dispatch.
    """
    path = os.environ.get("PCG_SV_SPLITTER", DEFAULT_SPLITTER)
    module_path, _, class_name = path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)
