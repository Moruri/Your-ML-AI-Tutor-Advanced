"""
Lesson 018 - Graphs as data

The bike-share network itself: docking stations joined by the streets riders
actually use, on two sides of a river with only a couple of bridges. We turn
it into the two matrices every graph method starts from (an adjacency matrix
and a node-feature matrix), then compute degrees, shortest paths and four
centralities by hand, checking the hand-written versions against scipy.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/network.png`, next to
this script (ignored by git), and runs in a few seconds.
"""

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def station_network(seed=18):
    """84 stations. 78 sit on two banks of a river (x < 0 west, x > 0 east);
    streets join stations closer than 1.15 km on the same bank, and three
    bridges cross the river. Six more stations form a new suburb whose bike
    lane doesn't reach the city yet. Returns coordinates, an edge list with
    travel minutes, and per-station features."""
    rng = np.random.default_rng(seed)
    west = np.column_stack([rng.uniform(-5, -0.4, 40), rng.uniform(0, 6, 40)])
    east = np.column_stack([rng.uniform(0.4, 4.5, 38), rng.uniform(0, 6, 38)])
    suburb = np.column_stack([rng.uniform(6.5, 7.6, 6), rng.uniform(4.6, 5.8, 6)])
    xy = np.vstack([west, east, suburb])
    n = len(xy)
    bank = np.r_[np.zeros(40, int), np.ones(38, int), np.full(6, 2)]
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    edges = set()
    for a in range(n):
        for b in range(a + 1, n):
            if bank[a] == bank[b] and D[a, b] < 1.15:
                edges.add((a, b))
    # make sure each bank is connected internally: join each station to its nearest same-bank neighbour
    for a in range(n):
        same = np.where((bank == bank[a]) & (np.arange(n) != a))[0]
        b = same[np.argmin(D[a, same])]
        edges.add((min(a, b), max(a, b)))
    for k in range(3):  # stitch any leftover islands on a bank to the rest of it
        while True:
            lab = connected_components(_adj(edges, n), directed=False)[1]
            groups = [g for g in range(lab.max() + 1) if np.isin(np.where(lab == g)[0], np.where(bank == k)[0]).all()]
            if len(groups) <= 1:
                break
            g0 = np.where(lab == groups[0])[0]
            rest = np.where((bank == k) & (lab != groups[0]))[0]
            a, b = np.unravel_index(np.argmin(D[np.ix_(g0, rest)]), (len(g0), len(rest)))
            edges.add((min(g0[a], rest[b]), max(g0[a], rest[b])))
    # three bridges across the river, near y = 1, 3 and 5
    for y in (1.0, 3.0, 5.0):
        w = np.argmin(np.abs(west[:, 0] + 0.4) + np.abs(west[:, 1] - y))
        e = 40 + np.argmin(np.abs(east[:, 0] - 0.4) + np.abs(east[:, 1] - y))
        edges.add((min(w, e), max(w, e)))
    edges = sorted(edges)
    minutes = np.array([D[a, b] * 4.0 * rng.uniform(1.0, 1.4) for a, b in edges])  # ~15 km/h, plus hills and lights
    # Node features: docks, height above the river (m), share of rides that are commutes, near a railway station
    docks = rng.choice([12, 16, 20, 24, 30], n)
    height = np.clip(12 * np.abs(xy[:, 0]) + rng.normal(0, 6, n), 0, None)
    near_rail = (np.abs(xy[:, 1] - 3.0) < 0.6).astype(int)
    commute = np.clip(0.35 + 0.3 * near_rail + rng.normal(0, 0.08, n), 0, 1)
    X = np.column_stack([docks, height, commute, near_rail])
    return dict(xy=xy, bank=bank, edges=np.array(edges), minutes=minutes, X=X,
                feat=["docks", "height_m", "commute_share", "near_rail"])


def _adj(edges, n):
    e = np.array(sorted(edges))
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()


G = station_network()
E, xy, X = G["edges"], G["xy"], G["X"]
n, m = len(xy), len(E)

# ---------------------------------------------------------------------------
# 1. Two matrices: who's connected, and what each node is like
# ---------------------------------------------------------------------------
print("\n1. From an edge list to matrices")
show("stations / streets (undirected edges)", f"{n} / {m}")
show("first three edges (station, station, minutes)", ", ".join(f"({a},{b},{t:.1f})" for (a, b), t in zip(E[:3], G["minutes"])))
A = np.zeros((n, n), int)
A[E[:, 0], E[:, 1]] = 1
A[E[:, 1], E[:, 0]] = 1
T = np.zeros((n, n))
T[E[:, 0], E[:, 1]] = T[E[:, 1], E[:, 0]] = G["minutes"]
show("A is symmetric / zero diagonal", f"{(A == A.T).all()} / {(np.diag(A) == 0).all()}")
show("density: edges / possible edges", f"{m / (n * (n - 1) / 2):.1%}")
As = sparse.csr_matrix(A)
show("numbers stored: dense A / sparse A (CSR)", f"{A.size} / {As.nnz + len(As.indptr) + len(As.indices)}")
show("1M stations, ~6 streets each: dense vs sparse", f"{1e12 * 8 / 1e12:.0f} TB vs about {(2 * 3e6 * 2 + 1e6) * 8 / 1e6:.0f} MB")
show("feature matrix X: shape / columns", f"{X.shape} / {', '.join(G['feat'])}")

# ---------------------------------------------------------------------------
# 2. Degrees
# ---------------------------------------------------------------------------
print("\n2. Degrees")
deg = A.sum(1)
show("degree = A @ 1: min / median / max", f"{deg.min()} / {np.median(deg):.0f} / {deg.max()}")
show("sum of degrees = 2 x edges", f"{deg.sum()} = 2 x {m}")
show("degree histogram (1, 2, ..., 8+)", " ".join(str(int(((deg == d) if d < 8 else (deg >= 8)).sum())) for d in range(1, 9)))
show("busiest station by degree", f"{deg.argmax()} ({deg.max()} streets)")
show("mean neighbour degree vs mean degree", f"{(A @ deg / deg).mean():.2f} vs {deg.mean():.2f}")

# ---------------------------------------------------------------------------
# 3. Paths: hops, minutes, walks and components
# ---------------------------------------------------------------------------
print("\n3. Paths")


def bfs(A, s):
    """Breadth-first search: hops from s to every station (-1 if unreachable)."""
    dist = np.full(len(A), -1)
    dist[s] = 0
    q = deque([s])
    while q:
        v = q.popleft()
        for w in np.nonzero(A[v])[0]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist


H = np.array([bfs(A, s) for s in range(n)])
Hs = dijkstra(As, unweighted=True)
show("hand BFS agrees with scipy (all pairs)", f"{np.array_equal(np.where(H < 0, np.inf, H), Hs)}")
ncomp, lab = connected_components(As, directed=False)
show("connected components (sizes)", f"{ncomp} ({', '.join(str(s) for s in np.bincount(lab))})")
big = lab == np.bincount(lab).argmax()
Hb = H[np.ix_(big, big)]
show("main component: diameter / mean hops", f"{Hb.max()} / {Hb[Hb > 0].mean():.2f}")
Tm = dijkstra(sparse.csr_matrix(T), directed=False)
Tb = Tm[np.ix_(big, big)]
show("main component: longest / mean trip (minutes)", f"{Tb.max():.0f} / {Tb[Tb > 0].mean():.1f}")
west_east = Tm[np.ix_(G["bank"] == 0, G["bank"] == 1)]
show("mean trip west -> east vs within west (min)", f"{west_east.mean():.1f} vs {Tm[np.ix_(G['bank'] == 0, G['bank'] == 0)][Tm[np.ix_(G['bank'] == 0, G['bank'] == 0)] > 0].mean():.1f}")
A2 = A @ A
A3 = A2 @ A
show("walks of length 2 / 3 from station 0 to itself", f"{A2[0, 0]} / {A3[0, 0]}")
show("(A @ A)[i, i] = degree of i, for every i", f"{(np.diag(A2) == deg).all()}")
show("triangles in the network = trace(A^3) / 6", f"{np.trace(A3) // 6}")

# ---------------------------------------------------------------------------
# 4. Centrality: four ways to be important
# ---------------------------------------------------------------------------
print("\n4. Centralities (main component)")
idx = np.where(big)[0]
Ab = A[np.ix_(idx, idx)]
nb = len(idx)
deg_c = Ab.sum(1) / (nb - 1)
close = (nb - 1) / H[np.ix_(idx, idx)].sum(1)


def brandes(A):
    """Betweenness (Brandes, 2001): the share of shortest paths between other
    pairs that pass through each node. Unweighted, undirected."""
    n = len(A)
    nbrs = [np.nonzero(A[v])[0] for v in range(n)]
    B = np.zeros(n)
    for s in range(n):
        stack, pred = [], [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1
        dist = np.full(n, -1)
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in nbrs[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                B[w] += delta[w]
    return B / 2  # each pair counted from both ends


btw = brandes(Ab)
Hm = H[np.ix_(idx, idx)]
show("check: sum of betweenness = sum over pairs of (hops - 1)", f"{btw.sum():.1f} = {(Hm[np.triu_indices(nb, 1)] - 1).sum():.1f}")


def power(M, iters=500):
    x = np.ones(len(M)) / len(M)
    for _ in range(iters):
        x = M @ x
        x /= np.abs(x).sum()
    return x


eig = power(Ab + np.eye(nb))  # + I avoids flip-flopping on near-bipartite graphs; same eigenvector
lam_max = np.linalg.eigvalsh(Ab.astype(float))[-1]
show("eigenvector: power iteration vs numpy (top value)", f"{(eig @ Ab @ eig) / (eig @ eig):.4f} vs {lam_max:.4f}")
d = 0.85
Pm = Ab / Ab.sum(0, keepdims=True)  # column-stochastic: walk to a random neighbour
pr = np.ones(nb) / nb
for _ in range(200):
    pr = (1 - d) / nb + d * Pm @ pr
show("PageRank sums to 1 / corr with degree", f"{pr.sum():.3f} / {np.corrcoef(pr, deg_c)[0, 1]:.2f}")
bridge_ends = np.isin(idx, np.unique(E[(G['bank'][E[:, 0]] != G['bank'][E[:, 1]])].ravel()))
for name, c in [("degree", deg_c), ("closeness", close), ("betweenness", btw), ("eigenvector", eig), ("PageRank", pr)]:
    top5 = np.argsort(-c)[:5]
    show(f"top 5 by {name}", f"{', '.join(str(idx[t]) for t in top5)}   ({bridge_ends[top5].sum()} at a bridge)")
show("bridge stations: mean degree rank / betweenness rank",
     f"{np.argsort(np.argsort(-deg_c))[bridge_ends].mean():.0f} / {np.argsort(np.argsort(-btw))[bridge_ends].mean():.0f} of {nb}")
r = lambda v: np.argsort(np.argsort(v))
C = np.corrcoef([r(deg_c), r(close), r(btw), r(eig), r(pr)])
show("rank corr: degree~PageRank, degree~betweenness", f"{C[0, 4]:.2f}, {C[0, 2]:.2f}")
show("rank corr: eigenvector~closeness", f"{C[3, 1]:.2f}")

# ---------------------------------------------------------------------------
# 5. Neighbours as features, and why node order mustn't matter
# ---------------------------------------------------------------------------
print("\n5. Using the graph in a model")
Dinv = np.diag(1 / np.maximum(deg, 1))
Xn = Dinv @ A @ X  # mean of each station's neighbours' features
for j, f in enumerate(G["feat"]):
    show(f"corr(own, neighbours' mean): {f}", f"{np.corrcoef(X[:, j], Xn[:, j])[0, 1]:.2f}")
perm = np.random.default_rng(1).permutation(n)
Pp = np.eye(n, dtype=int)[perm]
A_p, X_p = Pp @ A @ Pp.T, Pp @ X
show("relabel stations: degrees just get permuted", f"{np.array_equal(A_p.sum(1), deg[perm])}")
show("relabel: neighbour means just get permuted", f"{np.allclose(np.diag(1 / np.maximum(A_p.sum(1), 1)) @ A_p @ X_p, Xn[perm])}")
show("relabel: eigenvalues unchanged", f"{np.allclose(np.linalg.eigvalsh(A_p.astype(float)), np.linalg.eigvalsh(A.astype(float)))}")
show("relabel: row 0 of A as a feature vector", f"was {A[0].sum()} ones at {np.nonzero(A[0])[0][:3]}..., now at {np.nonzero(A_p[np.where(perm == 0)[0][0]])[0][:3]}...")

# ---------------------------------------------------------------------------
# Figure: the network sized by betweenness, coloured by degree
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, b in E:
    ax[0].plot(xy[[a, b], 0], xy[[a, b], 1], color="lightgrey", lw=0.8, zorder=0)
size = np.full(n, 8.0)
size[idx] = 8 + 400 * btw / btw.max()
sc = ax[0].scatter(xy[:, 0], xy[:, 1], s=size, c=deg, cmap="viridis", zorder=1)
ax[0].axvline(0, color="tab:blue", lw=6, alpha=0.15)
fig.colorbar(sc, ax=ax[0], label="degree")
ax[0].set_title("Stations: size = betweenness, colour = degree")
ax[0].set_aspect("equal")
ax[1].scatter(deg_c, btw, s=14)
ax[1].scatter(deg_c[bridge_ends], btw[bridge_ends], s=40, facecolors="none", edgecolors="r", label="bridge stations")
ax[1].set_xlabel("degree centrality")
ax[1].set_ylabel("betweenness")
ax[1].set_title("Few streets, many shortest paths")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "network.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/network.png")
