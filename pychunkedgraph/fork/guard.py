"""Refusals for a fork's writes that would land in its source's storage."""

from ..graph.exceptions import Forbidden
from ..meshing.mesh_meta import MeshMeta, dynamic_dir_name


def refuse_on_fork(cg, action: str) -> None:
    """Raise `Forbidden` when `cg` is a fork whose dynamic dir is not its own
    `dynamic_<fork>`: a fork made before its block was copied names its source's."""
    if cg.client.marker is None:
        return
    own = "/" + dynamic_dir_name(cg.graph_id)
    meshed = "dir" in cg.meta.custom_data.get("mesh", {})
    if not (meshed and MeshMeta(cg).dynamic_path.endswith(own)):
        raise Forbidden(
            f"{cg.graph_id} is a fork with no mesh dir of its own: {action} is refused"
        )
