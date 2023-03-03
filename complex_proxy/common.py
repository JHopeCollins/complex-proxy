
from enum import IntEnum

# flags for real and imaginary parts
Part = IntEnum("Part", (("Real", 0), ("Imag", 1)))
re = Part.Real
im = Part.Imag


def _flatten_tree(root, is_leaf, get_children, container=tuple):
    """
    Return the recursively flattened tree below root in the order that the leafs appear in the traversal.

    :arg root: the current root node.
    :arg is_leaf: predicate on root returning True if root has no children.
    :arg get_children: unary op on root that returns an iterable of root's children if is_leaf(root) evaluates False.
    :arg container: the container type to return the flattened tree in.
    """
    if is_leaf(root):
        return container((root,))
    else:
        return container((leaf
                          for child in get_children(root)
                          for leaf in _flatten_tree(child, is_leaf, get_children)))
