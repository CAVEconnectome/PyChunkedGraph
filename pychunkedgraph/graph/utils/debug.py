"""Helper functions to visualize/debug tree hierarchy."""


def dfs_print(cg, node, limit=2):
    """
    `limit` stop at this layer
    """
    stack = [(node, 0)]
    while stack:
        node, indent = stack.pop()
        print("     |" * indent, node)
        children = cg.get_children(node)
        if cg.get_chunk_layer(children[0]) < limit:
            continue
        for c in children:
            stack.append((c, indent + 1))
