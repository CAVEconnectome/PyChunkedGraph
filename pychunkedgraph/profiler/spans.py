"""The spans of an edit, each named beside the function that bounds it.

`install` wraps every target so its calls are timed into the open profiler scope; a
call made outside a scope goes straight through. The measured code holds no span:
this table is the one place that says what is timed.

Rows follow an edit's stages, from the Flask handlers to the write. A helper that runs
once per node inside a loop has no row; its batch-level caller stands for it.
"""

import functools
import importlib
import inspect
import sys
from typing import Callable, NamedTuple, Optional

from pychunkedgraph import get_logger

from .main import active_scope

logger = get_logger(__name__)


class Span(NamedTuple):
    """One timed function: its span's name, and the function as `module:qualname`."""

    name: str
    target: str
    # A coarse span, a stage or a handler, also reads the process RSS at enter and exit.
    memory: bool = False
    # Called with the function's result once its span has closed.
    then: Optional[Callable] = None


def _time_client_of(cg) -> None:
    """Time the storage calls of the client this graph talks through."""
    wrap_client(type(cg.client))


def _note_operation(lock) -> None:
    """Give the open scope the id of the operation whose root lock was just taken."""
    scope = active_scope()
    if scope is not None:
        scope.operation_id = int(lock.operation_id)


_COMMON = "pychunkedgraph.app.common"
_HANDLERS = "pychunkedgraph.app.segmentation.common"
_UTILS = "pychunkedgraph.app.app_utils"
_CG = "pychunkedgraph.graph.chunkedgraph:ChunkedGraph"
_OP = "pychunkedgraph.graph.operation"
_LOCKS = "pychunkedgraph.graph.locks"
_EDITS = "pychunkedgraph.graph.edits"
_PARENTS = "pychunkedgraph.graph.edits:CreateParentNodes"
_STALE = "pychunkedgraph.graph.edges.stale"
_CUT = "pychunkedgraph.graph.cutting"
_SV_SPLIT = "pychunkedgraph.graph.sv_split.edits"
_CACHE = "pychunkedgraph.graph.cache:CacheService"

SPANS: tuple[Span, ...] = (
    # The Flask half of a request.
    Span("before_request", f"{_COMMON}:before_request", memory=True),
    Span("handle_merge", f"{_HANDLERS}:handle_merge", memory=True),
    Span("handle_split", f"{_HANDLERS}:handle_split", memory=True),
    Span("handle_undo", f"{_HANDLERS}:handle_undo", memory=True),
    Span("handle_redo", f"{_HANDLERS}:handle_redo", memory=True),
    Span("get_cg", f"{_UTILS}:get_cg", memory=True, then=_time_client_of),
    Span("sv_lookup", f"{_UTILS}:handle_supervoxel_id_lookup", memory=True),
    Span("publish", f"{_HANDLERS}:publish_edit", memory=True),
    Span("jsonify", f"{_UTILS}:jsonify_with_kwargs"),
    Span("after_request", f"{_COMMON}:after_request", memory=True),
    # Stages 1 and 2: the operation, its roots and their locks.
    Span("add_edges", f"{_CG}.add_edges", memory=True),
    Span("remove_edges", f"{_CG}.remove_edges", memory=True),
    Span("undo_operation", f"{_CG}.undo_operation", memory=True),
    Span("redo_operation", f"{_CG}.redo_operation", memory=True),
    Span("execute", f"{_OP}:GraphEditOperation.execute", memory=True),
    Span("execute", f"{_OP}:RedoOperation.execute", memory=True),
    Span("execute", f"{_OP}:UndoOperation.execute", memory=True),
    Span("root_ids", f"{_OP}:MergeOperation._update_root_ids"),
    Span("root_ids", f"{_OP}:SplitOperation._update_root_ids"),
    Span("root_ids", f"{_OP}:MulticutOperation._update_root_ids"),
    Span("root_ids", f"{_OP}:RedoOperation._update_root_ids"),
    Span("root_ids", f"{_OP}:UndoOperation._update_root_ids"),
    Span("lock", f"{_LOCKS}:RootLock.__enter__", memory=True, then=_note_operation),
    Span("unlock", f"{_LOCKS}:RootLock.__exit__"),
    Span("lock_indefinite", f"{_LOCKS}:IndefiniteRootLock.__enter__"),
    Span("unlock_indefinite", f"{_LOCKS}:IndefiniteRootLock.__exit__"),
    Span("lineage", "pychunkedgraph.graph.lineage:lineage_graph"),
    Span("node_timestamps", f"{_CG}.get_node_timestamps"),
    Span("earliest_timestamp", f"{_CG}.get_earliest_timestamp"),
    # Stage 3: the edges to add or remove.
    Span("apply", f"{_OP}:MergeOperation._apply", memory=True),
    Span("apply", f"{_OP}:SplitOperation._apply", memory=True),
    Span("apply", f"{_OP}:MulticutOperation._apply", memory=True),
    Span("apply", f"{_OP}:RedoOperation._apply", memory=True),
    Span("apply", f"{_OP}:UndoOperation._apply", memory=True),
    Span("merge_preprocess", f"{_EDITS}:merge_preprocess"),
    Span("check_fake_edges", f"{_EDITS}:check_fake_edges"),
    Span("activated_edges", "pychunkedgraph.graph.misc:get_activated_edges"),
    Span("edges_status", "pychunkedgraph.graph.edges.utils:get_edges_status"),
    Span("multicut", f"{_CUT}:run_multicut", memory=True),
    Span("mincut_graph", f"{_CUT}:LocalMincutGraph.__init__", memory=True),
    Span("mincut", f"{_CUT}:LocalMincutGraph.compute_mincut", memory=True),
    Span(
        "mincut_check",
        f"{_CUT}:LocalMincutGraph._sink_and_source_connectivity_sanity_check",
    ),
    Span("merge_cross_edges", f"{_CUT}:merge_cross_chunk_edges_graph_tool"),
    # Stage 3c: a supervoxel split, as a phase. No row reaches inside the splitter
    # package: it is timed whole, by the block the split's own code opens around it.
    Span("sv_split_plan", f"{_SV_SPLIT}:plan_sv_splits", memory=True),
    Span("sv_split", f"{_SV_SPLIT}:split_supervoxels", memory=True),
    Span("sv_split_one", f"{_SV_SPLIT}:split_supervoxel", memory=True),
    Span("sv_split_lineage", f"{_SV_SPLIT}:copy_parents_and_add_lineage"),
    Span(
        "sv_split_write",
        "pychunkedgraph.graph.ocdbt.main:write_seg_chunks",
        memory=True,
    ),
    Span("l2_lock", f"{_LOCKS}:L2ChunkLock.__enter__", memory=True),
    Span("l2_unlock", f"{_LOCKS}:L2ChunkLock.__exit__"),
    Span("l2_lock_indefinite", f"{_LOCKS}:IndefiniteL2ChunkLock.__enter__"),
    Span("l2_unlock_indefinite", f"{_LOCKS}:IndefiniteL2ChunkLock.__exit__"),
    # Stage 4: new L2 nodes.
    Span("l2_add", f"{_EDITS}:add_edges", memory=True),
    Span("l2_remove", f"{_EDITS}:remove_edges", memory=True),
    Span("old_hierarchy", f"{_EDITS}:_init_old_hierarchy"),
    Span("affected_edges", f"{_EDITS}:_analyze_affected_edges"),
    Span("split_l2", f"{_EDITS}:_split_l2_agglomeration"),
    # Stage 5: the hierarchy above them.
    Span("rebuild", f"{_PARENTS}.run", memory=True),
    Span("components", f"{_PARENTS}._get_connected_components"),
    Span("cross_edge_dicts", f"{_PARENTS}._get_cross_edges_val_dicts"),
    Span("new_ids", f"{_PARENTS}._get_new_ids"),
    Span("layer_node_ids", f"{_PARENTS}._get_layer_node_ids"),
    Span("id_lineage", f"{_PARENTS}._update_id_lineage"),
    Span("root_lineage", f"{_PARENTS}._update_root_id_lineage"),
    Span("new_entries", f"{_PARENTS}.create_new_entries", memory=True),
    Span("stale_nodes", f"{_STALE}:get_stale_nodes"),
    Span("new_nodes", f"{_STALE}:get_new_nodes"),
    Span("descendants", f"{_EDITS}:_get_descendants_batch"),
    Span("counterparts", f"{_EDITS}:_get_counterparts"),
    Span("sanity_check", "pychunkedgraph.debug.utils:sanity_check"),
    # Stage 6: the write, and a failure's dump.
    Span("write", f"{_OP}:GraphEditOperation._write", memory=True),
    Span("err_dump", "pychunkedgraph.graph.err_dump:dump_err_artifact"),
    # Readers, timed under whichever stage calls them.
    Span("get_parents", f"{_CG}.get_parents"),
    Span("get_children", f"{_CG}.get_children"),
    Span("get_cross_chunk_edges", f"{_CG}.get_cross_chunk_edges"),
    Span("get_atomic_cross_edges", f"{_CG}.get_atomic_cross_edges"),
    Span("get_roots", f"{_CG}.get_roots"),
    Span("get_subgraph", f"{_CG}.get_subgraph", memory=True),
    Span("get_l2_agglomerations", f"{_CG}.get_l2_agglomerations"),
    Span("get_edges_from_edits", f"{_CG}.get_edges_from_edits"),
    Span("filter_stale_svs", f"{_CG}._filter_stale_svs"),
    Span("all_parents_dict", f"{_CG}.get_all_parents_dict_multiple"),
    Span("is_latest_roots", f"{_CG}.is_latest_roots"),
    Span("get_l2children", f"{_CG}.get_l2children"),
    Span("read_chunk_edges", f"{_CG}.read_chunk_edges"),
    Span("cache_parents", f"{_CACHE}.parents_multiple"),
    Span("cache_children", f"{_CACHE}.children_multiple"),
    Span("cache_cross_edges", f"{_CACHE}.cross_chunk_edges_multiple"),
    Span("get_chunk_edges", "pychunkedgraph.io.edges:get_chunk_edges"),
)

# The storage client's methods that reach storage, each timed as `bt.<method>`.
CLIENT_CALLS: tuple[str, ...] = (
    "read_nodes",
    "read_node",
    "write",
    "read_log_entry",
    "read_log_entries",
    "read_table_meta",
    "create_node_ids",
    "create_operation_id",
    "get_max_node_id",
    "get_max_operation_id",
    "get_consolidated_lock_timestamp",
    "lock_roots",
    "lock_roots_indefinitely",
    "lock_by_row_key",
    "lock_by_row_key_indefinitely",
    "lock_by_row_key_with_indefinite",
    "renew_locks",
    "renew_lock_by_row_key",
    "unlock_root",
    "unlock_indefinitely_locked_root",
    "unlock_by_row_key",
    "unlock_indefinitely_locked_by_row_key",
)


class _Patch(NamedTuple):
    """One replaced attribute; `original` is None when its owner only inherited it."""

    owner: object
    attr: str
    original: Optional[object]


_patches: list[_Patch] = []


def timed(span: Span, fn: Callable) -> Callable:
    """`fn`, timed as `span` into the open scope, and called as it is when none is open.

    The span's own bookkeeping never fails or alters the call: a raise while opening,
    closing or following up drops that step and is logged.
    """
    name, memory, then = span.name, span.memory, span.then

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        scope = active_scope()
        if scope is None:
            return fn(*args, **kwargs)
        try:
            entered = scope.enter(name, memory)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.warning(f"span {name}: not opened: {err}")
            return fn(*args, **kwargs)
        try:
            result = fn(*args, **kwargs)
        finally:
            try:
                scope.leave(name, entered)
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning(f"span {name}: not closed: {err}")
        if then is not None:
            try:
                then(result)
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning(f"span {name}: its follow-up failed: {err}")
        return result

    wrapper.pcg_span = name
    return wrapper


def resolve(target: str) -> tuple[object, str]:
    """The module or class holding a target, and the target's name in it."""
    module_name, _, qualname = target.partition(":")
    owner = importlib.import_module(module_name)
    *parents, attr = qualname.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    return owner, attr


def _replace(owner: object, attr: str, wrapper: object) -> None:
    _patches.append(_Patch(owner, attr, vars(owner).get(attr)))
    setattr(owner, attr, wrapper)


def _rebind(original: Callable, wrapper: Callable) -> None:
    """Point every `pychunkedgraph` module global that is `original` at `wrapper`: a
    module that imported the function by name holds a reference of its own."""
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("pychunkedgraph"):
            continue
        for key, value in list(vars(module).items()):
            if value is original:
                _replace(module, key, wrapper)


def install() -> int:
    """Wrap every `SPANS` target not wrapped yet. Returns how many attributes this
    call replaced, by-name references to a target included."""
    before = len(_patches)
    for span in SPANS:
        owner, attr = resolve(span.target)
        original = vars(owner)[attr]
        # A static or class method is wrapped through the function it holds.
        fn = getattr(original, "__func__", original)
        if hasattr(fn, "pcg_span"):
            continue
        wrapper = timed(span, fn)
        if fn is not original:
            wrapper = type(original)(wrapper)
        _replace(owner, attr, wrapper)
        if inspect.ismodule(owner):
            _rebind(original, wrapper)
    return len(_patches) - before


def wrap_client(client_type: type) -> None:
    """Time the storage calls of one client class, as `bt.<method>`. A call already
    timed, or one the class does not have, is left as it is."""
    for name in CLIENT_CALLS:
        method = getattr(client_type, name, None)
        if method is None or hasattr(method, "pcg_span"):
            continue
        target = f"{client_type.__module__}:{client_type.__qualname__}.{name}"
        _replace(client_type, name, timed(Span(f"bt.{name}", target), method))


def uninstall() -> None:
    """Put every original back, newest first."""
    while _patches:
        owner, attr, original = _patches.pop()
        if original is None:
            delattr(owner, attr)
        else:
            setattr(owner, attr, original)
