# HBase Client Implementation Summary

## Overview
This document summarizes the HBase-backed client implementation for PyChunkedGraph, providing an alternative to the BigTable backend.

## Implementation Status

### ✅ Completed

1. **HBase Client Implementation** (`pychunkedgraph/graph/client/hbase/client.py`)
   - Full implementation of all required methods from `ClientWithIDGen` and `OperationLogger`
   - Support for non-contiguous row set reads using HBase batch `Get` operations
   - Range scans for contiguous row reads
   - Locking mechanisms using HBase check-and-put patterns
   - ID generation using HBase atomic increments
   - Time-based filtering and column filtering

2. **Configuration System** (`pychunkedgraph/graph/client/hbase/__init__.py`)
   - `HBaseConfig` namedtuple similar to `BigTableConfig`
   - Environment variable support (HBASE_HOST, HBASE_PORT, etc.)
   - Helper function `get_client_info()` for easy configuration

3. **Client Factory** (`pychunkedgraph/graph/client/__init__.py`)
   - `get_client_from_info()` factory function that supports both "bigtable" and "hbase" backends
   - Updated `get_default_client_info()` to support backend selection via `BACKEND_TYPE` env var

4. **Bootstrap Integration** (`pychunkedgraph/ingest/utils.py`)
   - Updated to support both BigTable and HBase configurations
   - Automatic detection of backend type from config

5. **ChunkedGraph Integration** (`pychunkedgraph/graph/chunkedgraph.py`)
   - Updated to use the factory function instead of hardcoded BigTableClient

6. **Test Infrastructure** (`pychunkedgraph/tests/helpers.py`)
   - HBase server fixture using Docker
   - Updated `gen_graph` fixture to support backend selection
   - Helper functions for HBase emulator setup

7. **Utilities** (`pychunkedgraph/graph/client/hbase/utils.py`)
   - Helper functions for HBase-specific operations
   - Cell conversion utilities
   - Time range and column filtering

## Key Features

### Non-Contiguous Row Set Reading
HBase supports reading non-contiguous rows efficiently using batch `Get` operations:
- Implemented in `_read_byte_rows()` method
- Uses `table.rows()` for batch gets
- Supports parallel execution with multithreading
- Handles large request sets by batching

### Backend Selection
Users can select the backend via:
1. Environment variable: `BACKEND_TYPE=hbase` or `BACKEND_TYPE=bigtable`
2. Configuration dict: `{"backend_client": {"TYPE": "hbase", "CONFIG": {...}}}`

## Testing

### Setup
1. Virtual environment is set up using `uv`:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ```

2. Install dependencies (some may need to be installed manually due to build issues):
   ```bash
   uv pip install -e . --no-deps
   uv pip install pytest happybase numpy google-cloud-bigtable networkx pandas fastremap zstandard click pyyaml multiwrapper protobuf requests grpcio
   # Add other dependencies as needed
   ```

3. Run tests:
   ```bash
   source .venv/bin/activate
   python -m pytest pychunkedgraph/tests/ -v
   ```

   Or use the helper script:
   ```bash
   ./run_tests.sh pychunkedgraph/tests/ -v
   ```

### HBase Server for Testing
The test infrastructure includes a Docker-based HBase server fixture:
- Uses `harisekhon/hbase:latest` Docker image
- Automatically starts/stops for test sessions
- Exposes ports: 9090 (Thrift), 16000 (Master UI), 16010 (RegionServer UI)

### Running Tests with Both Backends
Tests can be parametrized to run against both backends:
```python
@pytest.mark.parametrize("backend", ["bigtable", "hbase"])
def test_something(gen_graph, backend):
    graph = gen_graph(backend=backend)
    # test code
```

## Configuration Example

### HBase Configuration
```python
config = {
    "backend_client": {
        "TYPE": "hbase",
        "CONFIG": {
            "HOST": "localhost",
            "PORT": 9090,
            "ADMIN": True,
            "READ_ONLY": False,
            "MAX_ROW_KEY_COUNT": 1000,
            "THRIFT_TRANSPORT": "buffered",
        }
    }
}
```

### BigTable Configuration (for comparison)
```python
config = {
    "backend_client": {
        "TYPE": "bigtable",
        "CONFIG": {
            "PROJECT": "my-project",
            "INSTANCE": "my-instance",
            "ADMIN": True,
            "READ_ONLY": False,
            "MAX_ROW_KEY_COUNT": 1000,
        }
    }
}
```

## Differences from BigTable Implementation

1. **Connection Management**: HBase uses persistent connections via `happybase.Connection`
2. **Locking**: Uses read-modify-write patterns instead of conditional mutations (HBase doesn't have exact equivalent to BigTable conditional rows)
3. **ID Generation**: Uses simple increment operations instead of append-based increments
4. **Batch Operations**: Uses `table.rows()` for batch gets and `table.batch()` for batch writes
5. **Time Filtering**: HBase uses millisecond timestamps vs BigTable's datetime objects

## Known Limitations

1. **Atomic Operations**: HBase's check-and-put is less atomic than BigTable's conditional mutations. Some race conditions may be possible in high-concurrency scenarios.
2. **Timestamp Precision**: HBase uses millisecond precision; BigTable uses microsecond precision (though both are rounded to milliseconds in practice).
3. **Dependencies**: Some dependencies (like `greenlet`) may have build issues on certain platforms. These can be worked around by installing pre-built wheels or using conda.

## Next Steps

1. Install remaining dependencies (cloudvolume, etc.) as needed for full test suite
2. Run full test suite against both backends
3. Fix any implementation issues discovered during testing
4. Add performance benchmarks comparing HBase vs BigTable
5. Document any HBase-specific optimizations or considerations

## Files Modified/Created

### New Files
- `pychunkedgraph/graph/client/hbase/client.py` - Main HBase client implementation
- `pychunkedgraph/graph/client/hbase/__init__.py` - Configuration (already existed, updated)
- `pychunkedgraph/graph/client/hbase/utils.py` - Utilities (already existed, updated)
- `run_tests.sh` - Test runner script
- `HBASE_IMPLEMENTATION.md` - This file

### Modified Files
- `pychunkedgraph/graph/client/__init__.py` - Added factory function and HBase support
- `pychunkedgraph/graph/chunkedgraph.py` - Updated to use factory function
- `pychunkedgraph/ingest/utils.py` - Added HBase config support
- `pychunkedgraph/tests/helpers.py` - Added HBase test infrastructure
- `requirements.in` - Added happybase dependency






