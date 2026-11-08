"""Interaction graphs: the undirected cycle C_n, the complete graph K_n, and a random
d-regular graph (extra arm 8, docs/handoff-extra-arms.md section 3).

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


def random_regular(n: int, d: int, seed: int, max_tries: int = 100_000) -> Graph:
    """A simple, connected d-regular graph on n agents, uniform over such graphs.

    Pairing (configuration) model with rejection: shuffle n * d half-edges with
    numpy.random.Generator(PCG64(seed)), pair them in order, and start again if the
    result has a self-loop, a repeated edge, or is not connected. Accepting only simple
    pairings makes the draw uniform over simple d-regular graphs; the connectivity
    condition then keeps it uniform over connected ones. The same seed gives the same
    graph. Needs n * d even and d < n.
    """
    import numpy as np

    if d < 1 or d >= n or (n * d) % 2:
        raise ValueError(f"no simple {d}-regular graph on {n} vertices")
    rng = np.random.Generator(np.random.PCG64(seed))
    stubs = np.repeat(np.arange(n), d)
    for _ in range(max_tries):
        pairs = rng.permutation(stubs).reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        e = {(int(min(a, b)), int(max(a, b))) for a, b in pairs}
        if len(e) != len(pairs):
            continue
        nb: list[list[int]] = [[] for _ in range(n)]
        for a, b in sorted(e):
            nb[a].append(b)
            nb[b].append(a)
        g = Graph(f"random_regular{n}_d{d}_seed{seed}", n, tuple(tuple(sorted(x)) for x in nb))
        if _connected(g):
            return g
    raise RuntimeError(f"no simple connected {d}-regular graph on {n} in {max_tries} tries")


def _connected(g: Graph) -> bool:
    seen, frontier = {0}, [0]
    while frontier:
        nxt = []
        for v in frontier:
            for w in g.in_neighbours[v]:
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return len(seen) == g.n


def diameter(g: Graph) -> int:
    """The largest graph distance between two agents (breadth-first search from each)."""
    best = 0
    for s in range(g.n):
        dist, frontier = {s: 0}, [s]
        while frontier:
            nxt = []
            for v in frontier:
                for w in g.in_neighbours[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        nxt.append(w)
            frontier = nxt
        if len(dist) != g.n:
            raise ValueError(f"{g.name} is not connected")
        best = max(best, max(dist.values()))
    return best


def make(kind: str, n: int, **kw) -> Graph:
    """A graph by kind. "random_regular" needs d= and seed= (extra arm 8)."""
    if kind == "cycle":
        return cycle(n)
    if kind == "complete":
        return complete(n)
    if kind == "random_regular":
        return random_regular(n, int(kw["d"]), int(kw["seed"]))
    raise ValueError(f"unknown graph kind {kind!r} (cycle, complete or random_regular)")
