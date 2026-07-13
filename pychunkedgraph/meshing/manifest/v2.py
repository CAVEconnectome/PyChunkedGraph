"""v2 graphene manifest: Accept-header negotiation + format assembly.

Branch-agnostic, pure-stdlib. The v2 format keys ``fragments`` by absolute mesh
bucket path so mesh location travels in the manifest itself; a client advertises
support via ``Accept: application/x.cave;manifest_version=2``.
"""

ACCEPT_MEDIA_TYPE = "application/x.cave"
MANIFEST_VERSION = 2  # highest manifest version this server produces


def requested_manifest_version(accept_header, default: int = 1) -> int:
    """Manifest version the client advertised via
    ``Accept: application/x.cave;manifest_version=N``.

    Parsed with stdlib — werkzeug's ``MIMEAccept`` mangles media-type params.
    """
    if not accept_header:
        return default
    for part in accept_header.split(","):
        params = [p.strip() for p in part.split(";")]
        if params[0].lower() != ACCEPT_MEDIA_TYPE:
            continue
        for param in params[1:]:
            key, _, value = param.partition("=")
            if key.strip().lower() == "manifest_version":
                try:
                    return int(value.strip())
                except ValueError:
                    return default
    return default


def to_v2_groups(fragments):
    """Split v1 fragments into ``(initial, dynamic)`` by the leading ``~`` marker.

    Fragments are kept verbatim: the ``~`` marks a sharded (byte-range) read vs a
    whole-file read, so it is the client's per-fragment dispatch flag and must
    survive into the v2 groups unchanged.
    """
    initial, dynamic = [], []
    for frag in fragments:
        (initial if frag.startswith("~") else dynamic).append(frag)
    return initial, dynamic


def assemble(initial_path: str, dynamic_path: str, initial_frags, dynamic_frags) -> dict:
    """v2 manifest dict: fragment lists grouped by absolute bucket path.

    Empty groups are omitted; each bucket value is a sub-object so per-bucket
    metadata can be added later without breaking the shape.
    """
    fragments = {}
    if len(initial_frags):
        fragments[initial_path] = {"fragments": list(initial_frags)}
    if len(dynamic_frags):
        fragments[dynamic_path] = {"fragments": list(dynamic_frags)}
    return {"manifest_version": MANIFEST_VERSION, "fragments": fragments}
