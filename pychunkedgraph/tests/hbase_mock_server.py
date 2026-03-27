"""In-process mock HBase REST (Stargate) server for testing.

Implements the subset of the HBase REST API used by
``pychunkedgraph.graph.client.hbase.client.Client``.
Runs in a daemon thread on a random port using stdlib ``http.server``.
"""

import base64
import json
import struct
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ScannerState:
    __slots__ = ("rows", "position", "batch_size")

    def __init__(self, rows, batch_size):
        self.rows = rows  # list of (row_key_bytes, cells_dict)
        self.position = 0
        self.batch_size = batch_size


class HBaseMockData:
    """Thread-safe shared state for the mock server."""

    def __init__(self):
        self.lock = threading.Lock()
        # tables[table_name][row_key: bytes][col_spec: str] = [(value: bytes, ts_ms: int), ...]
        self.tables: dict = {}
        self.table_schemas: dict = {}
        self.scanners: dict = {}
        self._scanner_counter = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64enc(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64dec(s: str) -> bytes:
    return base64.b64decode(s)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _insert_cell(cell_list: list, value: bytes, ts_ms: int):
    """Insert (value, ts_ms) keeping descending ts order."""
    entry = (value, ts_ms)
    for i, (_, t) in enumerate(cell_list):
        if ts_ms >= t:
            cell_list.insert(i, entry)
            return
    cell_list.append(entry)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


def _make_handler_class(data: HBaseMockData):
    """Create a handler class bound to the given shared data."""

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, format, *args):
            pass  # silence per-request logging

        # -- routing helpers ------------------------------------------------

        def _parse(self):
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            query = parse_qs(parsed.query)
            # flatten single-valued params
            qflat = {k: v[0] if len(v) == 1 else v for k, v in query.items()}
            return parts, qflat

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _json_body(self):
            raw = self._read_body()
            return json.loads(raw) if raw else {}

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_empty(self, code=200):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_bytes(
            self, raw: bytes, code=200, content_type="application/octet-stream"
        ):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        # -- GET ------------------------------------------------------------

        def do_GET(self):
            parts, query = self._parse()
            if len(parts) < 2:
                return self._send_empty(404)

            table = parts[0]

            # GET /{table}/schema
            if parts[1] == "schema":
                with data.lock:
                    if table in data.table_schemas:
                        return self._send_json(data.table_schemas[table])
                return self._send_empty(404)

            # GET /{table}/scanner/{id}
            if parts[1] == "scanner" and len(parts) >= 3:
                scanner_id = parts[2]
                with data.lock:
                    sc = data.scanners.get(scanner_id)
                    if sc is None:
                        return self._send_empty(404)
                    if sc.position >= len(sc.rows):
                        return self._send_empty(204)
                    end = sc.position + sc.batch_size
                    batch = sc.rows[sc.position : end]
                    sc.position = end
                return self._send_json(self._rows_to_cellset(batch))

            # GET /{table}/{key_b64}[/{col_spec}]
            row_key = _b64dec(parts[1])
            col_specs = parts[2].split(",") if len(parts) >= 3 else None
            max_versions = int(query.get("v", "1"))
            ts_from = int(query["ts.from"]) if "ts.from" in query else None
            ts_to = int(query["ts.to"]) if "ts.to" in query else None

            with data.lock:
                tbl = data.tables.get(table)
                if tbl is None or row_key not in tbl:
                    return self._send_empty(404)
                row_data = tbl[row_key]
                cells = self._filter_cells(
                    row_data, col_specs, ts_from, ts_to, max_versions
                )
                if not cells:
                    return self._send_empty(404)
            return self._send_json(self._rows_to_cellset([(row_key, cells)]))

        # -- PUT ------------------------------------------------------------

        def do_PUT(self):
            parts, query = self._parse()
            if len(parts) < 2:
                return self._send_empty(400)

            table = parts[0]

            # PUT /{table}/schema
            if parts[1] == "schema":
                body = self._json_body()
                with data.lock:
                    data.table_schemas[table] = body
                    data.tables.setdefault(table, {})
                return self._send_empty(200)

            # PUT /{table}/scanner
            if parts[1] == "scanner":
                return self._handle_create_scanner(table)

            # check=put / check=delete
            if "check" in query:
                if query["check"] == "put":
                    return self._handle_check_and_put(table, parts, query)
                if query["check"] == "delete":
                    return self._handle_check_and_delete(table, parts, query)

            # Batch write (PUT /{table}/{any_key})
            body = self._json_body()
            with data.lock:
                tbl = data.tables.setdefault(table, {})
                for row in body.get("Row", []):
                    rk = _b64dec(row["key"])
                    row_dict = tbl.setdefault(rk, {})
                    for cell in row.get("Cell", []):
                        col = _b64dec(cell["column"]).decode("utf-8")
                        val = _b64dec(cell["$"])
                        ts = cell.get("timestamp", _now_ms())
                        cell_list = row_dict.setdefault(col, [])
                        _insert_cell(cell_list, val, ts)
            return self._send_empty(200)

        # -- POST (atomic increment) ---------------------------------------

        def do_POST(self):
            parts, _ = self._parse()
            if len(parts) < 3:
                return self._send_empty(400)
            table = parts[0]
            row_key = _b64dec(parts[1])
            col_spec = parts[2]

            body = self._json_body()
            # Extract increment value from CellSet body
            inc_val = 0
            for row in body.get("Row", []):
                for cell in row.get("Cell", []):
                    raw = _b64dec(cell["$"])
                    inc_val = struct.unpack(">q", raw)[0]
                    break
                break

            with data.lock:
                tbl = data.tables.setdefault(table, {})
                row_dict = tbl.setdefault(row_key, {})
                cell_list = row_dict.get(col_spec, [])
                current = 0
                if cell_list:
                    # latest value is first (newest)
                    try:
                        current = struct.unpack(">q", cell_list[0][0])[0]
                    except struct.error:
                        current = 0
                new_val = current + inc_val
                new_bytes = struct.pack(">q", new_val)
                ts = _now_ms()
                new_list = [(new_bytes, ts)]
                row_dict[col_spec] = new_list

            return self._send_bytes(new_bytes)

        # -- DELETE ---------------------------------------------------------

        def do_DELETE(self):
            parts, _ = self._parse()
            if len(parts) < 2:
                return self._send_empty(400)

            table = parts[0]

            # DELETE /{table}/schema
            if parts[1] == "schema":
                with data.lock:
                    data.tables.pop(table, None)
                    data.table_schemas.pop(table, None)
                return self._send_empty(200)

            # DELETE /{table}/scanner/{id}
            if parts[1] == "scanner" and len(parts) >= 3:
                with data.lock:
                    data.scanners.pop(parts[2], None)
                return self._send_empty(200)

            row_key = _b64dec(parts[1])

            with data.lock:
                tbl = data.tables.get(table)
                if tbl is None:
                    return self._send_empty(200)

                if len(parts) == 2:
                    # DELETE row
                    tbl.pop(row_key, None)
                elif len(parts) == 3:
                    # DELETE column
                    col_spec = parts[2]
                    row_dict = tbl.get(row_key, {})
                    row_dict.pop(col_spec, None)
                    if not row_dict:
                        tbl.pop(row_key, None)
                elif len(parts) >= 4:
                    # DELETE cell version
                    col_spec = parts[2]
                    ts_ms = int(parts[3])
                    row_dict = tbl.get(row_key, {})
                    cell_list = row_dict.get(col_spec, [])
                    row_dict[col_spec] = [(v, t) for v, t in cell_list if t != ts_ms]
                    if not row_dict.get(col_spec):
                        row_dict.pop(col_spec, None)
                    if not row_dict:
                        tbl.pop(row_key, None)

            return self._send_empty(200)

        # -- Scanner --------------------------------------------------------

        def _handle_create_scanner(self, table):
            body = self._json_body()
            start_row = _b64dec(body["startRow"]) if "startRow" in body else b""
            end_row = _b64dec(body["endRow"]) if "endRow" in body else None
            batch_size = body.get("batch", 100)
            col_filter = body.get("column")  # list of col_spec strings or None
            start_time = body.get("startTime")  # ms, inclusive
            end_time = body.get("endTime")  # ms, exclusive

            with data.lock:
                tbl = data.tables.get(table, {})
                filtered = []
                for rk in sorted(tbl.keys()):
                    if rk < start_row:
                        continue
                    if end_row is not None and rk >= end_row:
                        continue
                    cells = self._filter_cells(
                        tbl[rk], col_filter, start_time, end_time, max_versions=None
                    )
                    if cells:
                        filtered.append((rk, cells))

                data._scanner_counter += 1
                scanner_id = str(data._scanner_counter)
                data.scanners[scanner_id] = ScannerState(filtered, batch_size)

            host, port = self.server.server_address
            loc = f"http://{host}:{port}/{table}/scanner/{scanner_id}"
            self.send_response(201)
            self.send_header("Location", loc)
            self.send_header("Content-Length", "0")
            self.end_headers()

        # -- Check-and-put --------------------------------------------------

        def _handle_check_and_put(self, table, parts, query):
            body = self._json_body()
            row_key = _b64dec(parts[1])
            check_col_spec = parts[2]  # the column to check

            row_cells = body.get("Row", [{}])[0].get("Cell", [])

            if len(row_cells) >= 2:
                # First cell is the check cell, second is the put cell
                check_value = _b64dec(row_cells[0]["$"])
                put_cell = row_cells[1]
            else:
                # Single cell: check that column does NOT exist
                check_value = None
                put_cell = row_cells[0] if row_cells else None

            with data.lock:
                tbl = data.tables.setdefault(table, {})
                row_dict = tbl.get(row_key, {})
                current_cells = row_dict.get(check_col_spec, [])

                if check_value is None:
                    # Column must not exist
                    if current_cells:
                        return self._send_empty(304)
                else:
                    # Latest value must match
                    if not current_cells or current_cells[0][0] != check_value:
                        return self._send_empty(304)

                # Condition met - apply the put
                if put_cell:
                    put_col = _b64dec(put_cell["column"]).decode("utf-8")
                    put_val = _b64dec(put_cell["$"])
                    put_ts = put_cell.get("timestamp", _now_ms())
                    row_dict = tbl.setdefault(row_key, {})
                    cell_list = row_dict.setdefault(put_col, [])
                    _insert_cell(cell_list, put_val, put_ts)

            return self._send_empty(200)

        # -- Check-and-delete -----------------------------------------------

        def _handle_check_and_delete(self, table, parts, query):
            body = self._json_body()
            row_key = _b64dec(parts[1])
            check_col_spec = parts[2]

            row_cells = body.get("Row", [{}])[0].get("Cell", [])
            check_value = _b64dec(row_cells[0]["$"]) if row_cells else None

            with data.lock:
                tbl = data.tables.get(table, {})
                row_dict = tbl.get(row_key, {})
                current_cells = row_dict.get(check_col_spec, [])

                if not current_cells or current_cells[0][0] != check_value:
                    return self._send_empty(304)

                # Match - delete the column
                row_dict.pop(check_col_spec, None)
                if not row_dict:
                    tbl.pop(row_key, None)

            return self._send_empty(200)

        # -- Utility --------------------------------------------------------

        def _filter_cells(self, row_data, col_specs, ts_from, ts_to, max_versions):
            """Filter a row's cells by column specs and time range."""
            result = {}
            for col, cell_list in row_data.items():
                if col_specs and col not in col_specs:
                    continue
                filtered = []
                for val, ts in cell_list:
                    if ts_from is not None and ts < ts_from:
                        continue
                    if ts_to is not None and ts >= ts_to:
                        continue
                    filtered.append((val, ts))
                if max_versions is not None:
                    filtered = filtered[:max_versions]
                if filtered:
                    result[col] = filtered
            return result

        def _rows_to_cellset(self, rows):
            """Convert list of (row_key_bytes, cells_dict) to CellSet JSON."""
            out_rows = []
            for rk, cells in rows:
                out_cells = []
                for col, cell_list in cells.items():
                    for val, ts in cell_list:
                        out_cells.append(
                            {
                                "column": _b64enc(col.encode("utf-8")),
                                "$": _b64enc(val),
                                "timestamp": ts,
                            }
                        )
                out_rows.append(
                    {
                        "key": _b64enc(rk),
                        "Cell": out_cells,
                    }
                )
            return {"Row": out_rows}

    return Handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_hbase_mock_server(host="127.0.0.1", port=0):
    """Start mock HBase REST server in a daemon thread.

    Returns ``(data, server, port)`` where *port* is the actual bound port.
    """
    mock_data = HBaseMockData()
    handler_cls = _make_handler_class(mock_data)
    server = ThreadingHTTPServer((host, port), handler_cls)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return mock_data, server, actual_port
