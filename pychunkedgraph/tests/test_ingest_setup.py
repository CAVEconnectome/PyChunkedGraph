"""Ingest setup is idempotent under `exist_ok` (the orchestrator's converge/resume re-run)."""

import uuid
from datetime import timedelta

import pytest
from google.auth import credentials

from .helpers import bigtable_emulator  # noqa: F401  (pytest fixture)
from ..pipeline.ingest import setup as setup_mod

_CONFIG = {
    "data_source": {
        "EDGES": "gs://chunked-graph/minnie65_0/edges",
        "COMPONENTS": "gs://chunked-graph/minnie65_0/components",
        "WATERSHED": "gs://microns-seunglab/minnie65/ws_minnie65_0",
    },
    "graph_config": {
        "CHUNK_SIZE": [512, 512, 64],
        "FANOUT": 2,
        "SPATIAL_BITS": 10,
        "ID_PREFIX": "",
        "ROOT_LOCK_EXPIRY": timedelta(seconds=5),
    },
    "backend_client": {
        "TYPE": "bigtable",
        "CONFIG": {
            "ADMIN": True,
            "READ_ONLY": False,
            "PROJECT": "IGNORE_ENVIRONMENT_PROJECT",
            "INSTANCE": "emulated_instance",
            "CREDENTIALS": credentials.AnonymousCredentials(),
            "MAX_ROW_KEY_COUNT": 1000,
        },
    },
    "ingest_config": {},
}


def test_setup_exist_ok_is_idempotent(bigtable_emulator, monkeypatch, tmp_path):
    # the real config holds non-YAML objects (credentials, timedelta); feed the dict directly
    monkeypatch.setattr(setup_mod.yaml, "safe_load", lambda stream: _CONFIG)
    dataset = tmp_path / "dataset.yml"
    dataset.write_text("{}")
    gid = f"test_{uuid.uuid4().hex[:8]}"

    setup_mod.setup(gid, dataset_path=str(dataset))  # first run creates the table + meta
    setup_mod.setup(gid, exist_ok=True, dataset_path=str(dataset))  # re-run: no-op, no raise
    with pytest.raises(ValueError):
        setup_mod.setup(gid, dataset_path=str(dataset))  # without exist_ok, still errors
