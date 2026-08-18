"""Deterministic, coordinate-free spatial layout for graphs without GPS.

PEMS03, PEMS04 and PEMS08 ship no coordinates. The previous converter filled the
gap with the first two non-trivial Laplacian eigenvectors, which is not a
position field: on a disconnected graph those vectors lie inside the null space
and collapse to component indicators, the eigenbasis for a repeated eigenvalue
is arbitrary, and eigenvector signs are unconstrained -- so the "coordinates"
were neither geometric nor reproducible across machines.

This replaces them with a layout defined directly on the graph. It produces the
same object ALFTemporal actually needs -- an ordering of nodes into parts -- and
never pretends to be a coordinate system.

``LAYOUT_ALGORITHM`` / ``LAYOUT_VERSION`` are part of the graph contract hash, so
changing the traversal changes the contract rather than silently changing
results.

Determinism rules, in order of application:

1. The structural view is ``S = 1[(A + A.T) > 0]`` with the diagonal cleared, so
   direction and edge weights cannot affect the layout.
2. Components are ordered by ``(-size, smallest sensor id)``.
3. Within a component, traversal starts at ``(-degree, sensor id)`` and expands
   breadth-first, visiting each frontier's neighbours in ``(-degree, sensor id)``
   order.
4. Sensor id breaks every remaining tie.

Rule 4 matters and is stated rather than glossed: two nodes that are genuinely
interchangeable under graph isomorphism are separated by their canonical sensor
id, so the layout is deterministic given the id assignment. It is *not* invariant
to relabelling nodes arbitrarily -- only to relabelling that carries the ids
along with the nodes.
"""

from collections import deque

import numpy as np

LAYOUT_ALGORITHM = "graph_native_component_bfs"
LAYOUT_VERSION = 1


def structural_view(adjacency):
    """Undirected binary projection used only for layout and structure.

    Never substituted back into propagation: a symmetrised graph collapses
    forward and backward diffusion into one operator.
    """
    dense = np.asarray(adjacency, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(f"adjacency must be square, got {dense.shape}")
    view = ((dense + dense.T) > 0).astype(np.int8)
    np.fill_diagonal(view, 0)
    return view


def connected_components(view):
    """Components of the structural view, ordered by (-size, smallest id)."""
    num_nodes = view.shape[0]
    seen = np.zeros(num_nodes, dtype=bool)
    components = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbour in np.nonzero(view[node])[0]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(int(neighbour))
        components.append(sorted(int(m) for m in members))
    components.sort(key=lambda group: (-len(group), group[0]))
    return components


def node_order(adjacency):
    """A deterministic permutation of every node, components kept contiguous."""
    view = structural_view(adjacency)
    degree = view.sum(axis=1)
    order = []
    for component in connected_components(view):
        remaining = set(component)
        # Root: highest degree, then smallest sensor id.
        root = min(component, key=lambda node: (-int(degree[node]), node))
        queue = deque([root])
        remaining.discard(root)
        visited = [root]
        while queue:
            node = queue.popleft()
            neighbours = [int(n) for n in np.nonzero(view[node])[0] if n in remaining]
            neighbours.sort(key=lambda n: (-int(degree[n]), n))
            for neighbour in neighbours:
                remaining.discard(neighbour)
                visited.append(neighbour)
                queue.append(neighbour)
        # A component is traversed exhaustively, so `remaining` is empty here;
        # the guard exists so a future change cannot silently drop nodes.
        if remaining:
            raise AssertionError(f"component traversal missed {sorted(remaining)}")
        order.extend(visited)
    if sorted(order) != list(range(view.shape[0])):
        raise AssertionError("layout order is not a permutation of the nodes")
    return order


def layout_parts(adjacency, recur_times):
    """Split the deterministic order into ``2 ** recur_times`` contiguous parts.

    Contiguous on purpose: components are already contiguous in the order, so
    chunking keeps most parts inside a single component without inventing any
    edge between components.
    """
    order = node_order(adjacency)
    num_parts = 2 ** int(recur_times)
    if num_parts > len(order):
        raise ValueError(
            f"recur_times={recur_times} asks for {num_parts} parts but the graph "
            f"has only {len(order)} nodes")
    base, extra = divmod(len(order), num_parts)
    parts, cursor = [], 0
    for index in range(num_parts):
        size = base + (1 if index < extra else 0)
        parts.append(np.asarray(order[cursor:cursor + size], dtype=np.int64))
        cursor += size
    return parts


def layout_signature(adjacency, recur_times):
    """Canonical text of the layout, for hashing into the graph contract."""
    parts = layout_parts(adjacency, recur_times)
    lines = [f"{LAYOUT_ALGORITHM}/v{LAYOUT_VERSION}", f"recur_times={int(recur_times)}"]
    for index, part in enumerate(parts):
        lines.append(f"part{index}:" + ",".join(str(int(n)) for n in part))
    return "\n".join(lines)
