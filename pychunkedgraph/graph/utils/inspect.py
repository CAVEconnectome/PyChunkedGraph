"""Helper functions to visualize/inspect tree hierarchy."""


def dfs_print_node(cg, node, limit=2, sep="    |"):
    """
    `limit` stop at this layer
    """
    leaves = 0
    stack = [(node, 0)]
    while stack:
        node, indent = stack.pop()
        children = cg.get_children(node)
        print(sep * indent, node, f"({cg.get_chunk_layer(node)}, {len(children)})")
        if cg.get_chunk_layer(children[0]) < limit:
            leaves += len(children)
            continue
        for c in children:
            stack.append((c, indent + 1))
    print("leaves count", leaves)
