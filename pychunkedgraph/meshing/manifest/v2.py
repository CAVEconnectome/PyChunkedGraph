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


def to_v2_groups(node_ids, fragments, return_seg_ids):
    """Group v1 fragments into ``(initial, dynamic)``, each row plain under its bucket.

    A dynamic fragment is named after its node id, so ``return_seg_ids`` prefixes
    the sharded rows alone.
    """
    initial, dynamic = [], []
    for node_id, frag in zip(node_ids, fragments):
        if not frag.startswith("~"):
            dynamic.append(frag)
            continue
        shard = frag[1:]
        initial.append(f"{node_id}:{shard}" if return_seg_ids else shard)
    return initial, dynamic


def assemble(
    initial_path: str, dynamic_path: str, initial_frags, dynamic_frags
) -> dict[str, object]:
    """v2 manifest: fragment lists grouped by absolute bucket path.

    Empty groups are omitted. Metadata is added beside ``fragments`` at the top
    level, so a bucket maps straight to its list.
    """
    fragments: dict[str, list[str]] = {}
    if len(initial_frags):
        fragments[initial_path] = list(initial_frags)
    if len(dynamic_frags):
        fragments[dynamic_path] = list(dynamic_frags)
    return {"manifest_version": MANIFEST_VERSION, "fragments": fragments}
