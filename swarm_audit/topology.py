"""Interaction graphs: the undirected cycle C_n and the complete graph K_n.

An undirected graph is read as directed both ways (setup-paper section 2), so every
edge {i, j} gives i -> j and j -> i. Agent i's record counts the outputs of its
in-neighbours; on these graphs in- and out-neighbours coincide.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Graph:
    name: str
    n: int
    in_neighbours: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.in_neighbours) != self.n:
            raise ValueError("need one in-neighbour list per agent")
        for i, nb in enumerate(self.in_neighbours):
            if i in nb or len(set(nb)) != len(nb) or any(not 0 <= j < self.n for j in nb):
                raise ValueError(f"bad in-neighbours for agent {i}: {nb}")

    def in_degree(self, i: int) -> int:
        return len(self.in_neighbours[i])

    @cached_property
    def degrees(self) -> tuple[int, ...]:
        return tuple(len(nb) for nb in self.in_neighbours)

    @cached_property
    def regular_degree(self) -> int:
        """The common in-degree; raises if the graph is not regular."""
        ds = set(self.degrees)
        if len(ds) != 1:
            raise ValueError(f"{self.name} is not regular: degrees {sorted(ds)}")
        return ds.pop()

    @cached_property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Undirected edges (i, j) with i < j: pairs joined in either direction."""
        return tuple(sorted({(min(i, j), max(i, j))
                             for i, nb in enumerate(self.in_neighbours) for j in nb}))

    @cached_property
    def adjacency(self):
        """A[i, j] = 1 if j is an in-neighbour of i (numpy int array, n x n)."""
        import numpy as np

        a = np.zeros((self.n, self.n), dtype=np.int64)
        for i, nb in enumerate(self.in_neighbours):
            a[i, list(nb)] = 1
        return a

    def distance(self, i: int, j: int) -> int:
        """Graph distance (breadth-first search over in-neighbour links)."""
        if i == j:
            return 0
        seen, frontier, d = {i}, [i], 0
        while frontier:
            d += 1
            nxt = []
            for v in frontier:
                for w in self.in_neighbours[v]:
                    if w == j:
                        return d
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
            frontier = nxt
        raise ValueError(f"{j} is not reachable from {i}")


def cycle(n: int) -> Graph:
    """C_n: agent i's in-neighbours are i - 1 and i + 1 (mod n). Needs n >= 3."""
    if n < 3:
        raise ValueError("a cycle needs n >= 3")
    return Graph(f"cycle{n}", n, tuple(((i - 1) % n, (i + 1) % n) for i in range(n)))


def complete(n: int) -> Graph:
    """K_n: every other agent is an in-neighbour."""
    if n < 2:
        raise ValueError("a complete graph needs n >= 2")
    return Graph(f"complete{n}", n, tuple(tuple(j for j in range(n) if j != i)
                                          for i in range(n)))


def make(kind: str, n: int) -> Graph:
    if kind == "cycle":
        return cycle(n)
    if kind == "complete":
        return complete(n)
    raise ValueError(f"unknown graph kind {kind!r} (cycle or complete)")
