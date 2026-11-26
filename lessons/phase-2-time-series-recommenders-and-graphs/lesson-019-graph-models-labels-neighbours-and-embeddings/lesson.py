"""
Lesson 019 - Graph models: labels, neighbours and embeddings

The city has grown to 600 docking stations. The operator wants to know what
kind of place each station serves (office district, residential, or leisure)
because it decides when vans rebalance bikes. A survey only covered 60 of
them. Can the street network fill in the other 540?

We go from the simplest idea to a small graph neural network, all by hand:
neighbour votes, label propagation, a graph convolution as feature smoothing,
a two-layer GCN with hand-written gradients, and node embeddings learned from
random walks. Then we predict missing streets (link prediction), and finish
with the trap: a graph where neighbours are *unlike* each other.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/graph_models.png`, next
to this script (ignored by git), and runs in well under a minute on a laptop.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
CLASSES = ["office", "residential", "leisure"]


def show(label, value):
    print(f"  {label:<50} -> {value}")


def city(seed=19, n=600):
    """Stations scattered around 15 neighbourhood centres, each of one type.
    Streets join stations closer than a threshold, so the graph is local and
    neighbours tend to share a type (homophily). Features are noisy hints."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 10, (15, 2))
    ctype = np.array([0, 1, 2] * 5)
    home = rng.integers(0, 15, n)
    xy = centres[home] + rng.normal(0, 0.75, (n, 2))
    # a station's type is its nearest centre's type, but 8% are odd ones out
    near = np.argmin(((xy[:, None] - centres[None]) ** 2).sum(-1), 1)
    y = ctype[near].copy()
    odd = rng.random(n) < 0.08
    y[odd] = rng.integers(0, 3, odd.sum())
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    A = (D < 0.42).astype(float)
    nn = np.argsort(D, 1)[:, :2]  # nobody is left without at least two streets
    A[np.repeat(np.arange(n), 2), nn.ravel()] = 1
    A = np.maximum(A, A.T)
    # Features: morning-peak share, weekend share, docks, distance to centre of town.
    # The type shifts each one a little; the noise is big. That's the realistic part.
    morning = np.array([0.42, 0.30, 0.22])[y] + rng.normal(0, 0.11, n)
    weekend = np.array([0.18, 0.27, 0.36])[y] + rng.normal(0, 0.11, n)
    docks = rng.choice([12, 16, 20, 24, 30], n) + np.array([4, 0, -2])[y]
    to_centre = np.sqrt(((xy - 5) ** 2).sum(1))
    X = np.column_stack([morning, weekend, docks, to_centre])
    return xy, A, X, y


def sym_norm(A):
    """D^-1/2 (A + I) D^-1/2: the GCN propagation matrix (self-loops included)."""
    At = A + np.eye(len(A))
    d = At.sum(1)
    return At / np.sqrt(d)[:, None] / np.sqrt(d)[None]


def splits(n, k, reps, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        lab = np.zeros(n, bool)
        lab[rng.choice(n, k, replace=False)] = True
        yield lab


def logreg(Z, y, lab, C=1.0):
    sc = StandardScaler().fit(Z[lab])
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Z[lab]), y[lab])
    return m.predict(sc.transform(Z))


def acc(pred, y, mask):
    return (pred[mask] == y[mask]).mean()


xy, A, X, y = city()
n = len(y)
deg = A.sum(1)
E = np.array(np.nonzero(np.triu(A))).T
K_LAB, REPS = 60, 20

# ---------------------------------------------------------------------------
# 1. The problem: 60 labels, 600 stations, one network
# ---------------------------------------------------------------------------
print("\n1. The problem")
show("stations / streets / mean degree", f"{n} / {len(E)} / {deg.mean():.1f}")
ncomp, comp = sparse.csgraph.connected_components(sparse.csr_matrix(A), directed=False)
show("connected components (largest)", f"{ncomp} ({np.bincount(comp).max()})")
show("class counts (office, residential, leisure)", " / ".join(str(c) for c in np.bincount(y)))
h_edge = (y[E[:, 0]] == y[E[:, 1]]).mean()
show("edge homophily: streets joining the same type", f"{h_edge:.1%}  (chance would be {(np.bincount(y) / n) @ (np.bincount(y) / n):.1%})")
show("surveyed (labelled) stations per split / splits", f"{K_LAB} / {REPS}")

# ---------------------------------------------------------------------------
# 2. Baselines, then a vote among neighbours
# ---------------------------------------------------------------------------
print("\n2. Baselines and neighbour votes (accuracy on the 540 unlabelled, mean of 20 splits)")
res = {k: [] for k in ["majority", "features only", "1-hop vote", "1-hop coverage", "label propagation",
                       "features + LP scores", "SGC (A^2 X)", "GCN (2 layers)", "walk embedding",
                       "walk embedding + features", "spectral embedding"]}


def label_prop(S, Y0, alpha=0.9, iters=None):
    """Zhou et al. (2004): F = (1 - alpha) (I - alpha S)^-1 Y0, or iterate F <- alpha S F + (1 - alpha) Y0."""
    if iters is None:
        return (1 - alpha) * spsolve(sparse.csc_matrix(np.eye(len(S)) - alpha * S), Y0)
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * S @ F + (1 - alpha) * Y0
    return F


S_lp = A / np.sqrt(deg)[:, None] / np.sqrt(deg)[None]
first = True
for lab in splits(n, K_LAB, REPS):
    un = ~lab
    res["majority"].append(acc(np.full(n, np.bincount(y[lab]).argmax()), y, un))
    res["features only"].append(acc(logreg(X, y, lab), y, un))
    Y0 = np.zeros((n, 3))
    Y0[lab, y[lab]] = 1
    votes = A @ Y0
    has = votes.sum(1) > 0
    vote = np.where(has, votes.argmax(1), np.bincount(y[lab]).argmax())
    res["1-hop vote"].append(acc(vote, y, un))
    res["1-hop coverage"].append(has[un].mean())
    F = label_prop(S_lp, Y0)
    if first:
        F_it = label_prop(S_lp, Y0, iters=200)
        show("label propagation: closed form = 200 iterations", f"{np.allclose(F, F_it, atol=1e-6)}  (max gap {np.abs(F - F_it).max():.1e})")
        first = False
    res["label propagation"].append(acc(F.argmax(1), y, un))
    Fn = F / np.maximum(F.sum(1, keepdims=True), 1e-12)
    res["features + LP scores"].append(acc(logreg(np.column_stack([X, Fn]), y, lab), y, un))
for k in ["majority", "features only", "1-hop vote", "1-hop coverage", "label propagation", "features + LP scores"]:
    v = np.array(res[k])
    show(k, f"{v.mean():.1%}  (+/- {v.std():.1%})")

# ---------------------------------------------------------------------------
# 3. Graph convolution: smooth the features, then classify
# ---------------------------------------------------------------------------
print("\n3. Graph convolution")
Ahat = sym_norm(A)
Xs = StandardScaler().fit_transform(X)
for hops in [0, 1, 2, 3, 6, 20]:
    Z = np.linalg.matrix_power(Ahat, hops) @ Xs
    a = [acc(logreg(Z, y, lab), y, ~lab) for lab in splits(n, K_LAB, REPS)]
    if hops == 2:
        res["SGC (A^2 X)"] = a
    show(f"logistic regression on Ahat^{hops} X", f"{np.mean(a):.1%}")
Z20 = np.linalg.matrix_power(Ahat, 20) @ Xs
Z0 = Xs
show("spread of a feature across stations: hops 0 vs 20", f"{Z0[:, 0].std():.2f} vs {Z20[:, 0].std():.2f}  (over-smoothing)")


def gcn_train(Ahat, Xs, y, lab, hidden=16, epochs=300, lr=0.05, wd=5e-3, drop=0.5, seed=0):
    """Two-layer GCN (Kipf & Welling, 2017): softmax(Ahat relu(Ahat X W1) W2).
    Full-batch gradient descent with Adam, gradients written out by hand."""
    rng = np.random.default_rng(seed)
    d, c = Xs.shape[1], 3
    W1 = rng.normal(0, np.sqrt(2 / (d + hidden)), (d, hidden))
    W2 = rng.normal(0, np.sqrt(2 / (hidden + c)), (hidden, c))
    P = [W1, W2]
    m = [np.zeros_like(w) for w in P]
    v = [np.zeros_like(w) for w in P]
    AX = Ahat @ Xs  # fixed: compute once
    Y = np.eye(c)[y]
    L = lab.sum()
    losses = []
    for t in range(1, epochs + 1):
        H_pre = AX @ W1
        H = np.maximum(H_pre, 0)
        mask = (rng.random(H.shape) > drop) / (1 - drop)
        Hd = H * mask
        O = Ahat @ Hd @ W2
        O -= O.max(1, keepdims=True)
        Pr = np.exp(O)
        Pr /= Pr.sum(1, keepdims=True)
        loss = -np.log(Pr[lab, y[lab]] + 1e-12).mean()
        losses.append(loss)
        G = np.zeros_like(Pr)
        G[lab] = (Pr[lab] - Y[lab]) / L          # dLoss/dO, only labelled rows count
        AH = Ahat @ Hd
        gW2 = AH.T @ G + wd * W2
        gHd = Ahat.T @ G @ W2.T
        gH = gHd * mask * (H_pre > 0)
        gW1 = AX.T @ gH + wd * W1
        for i, g in enumerate([gW1, gW2]):        # Adam
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            P[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    O = Ahat @ np.maximum(AX @ W1, 0) @ W2
    return O.argmax(1), losses, (W1, W2)


def gcn_grad_check(Ahat, Xs, y, lab, seed=3):
    """Compare the hand-written gradient for one weight with a finite difference."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, (Xs.shape[1], 8))
    W2 = rng.normal(0, 0.5, (8, 3))
    AX = Ahat @ Xs

    def loss(W1, W2):
        O = Ahat @ np.maximum(AX @ W1, 0) @ W2
        O -= O.max(1, keepdims=True)
        Pr = np.exp(O) / np.exp(O).sum(1, keepdims=True)
        return -np.log(Pr[lab, y[lab]]).mean(), Pr

    l0, Pr = loss(W1, W2)
    G = np.zeros_like(Pr)
    G[lab] = (Pr[lab] - np.eye(3)[y[lab]]) / lab.sum()
    H_pre = AX @ W1
    gH = (Ahat.T @ G @ W2.T) * (H_pre > 0)
    g = (AX.T @ gH)[1, 2]
    W1b = W1.copy()
    W1b[1, 2] += 1e-6
    return g, (loss(W1b, W2)[0] - l0) / 1e-6


g_hand, g_num = gcn_grad_check(Ahat, Xs, y, next(splits(n, K_LAB, 1)))
show("GCN gradient check: hand vs finite difference", f"{g_hand:.6f} vs {g_num:.6f}")
for i, lab in enumerate(splits(n, K_LAB, REPS)):
    pred, losses, _ = gcn_train(Ahat, Xs, y, lab, seed=i)
    res["GCN (2 layers)"].append(acc(pred, y, ~lab))
    if i == 0:
        losses0 = losses
show("GCN training loss: epoch 1 / 300", f"{losses0[0]:.3f} / {losses0[-1]:.3f}")
show("GCN accuracy on unlabelled", f"{np.mean(res['GCN (2 layers)']):.1%}  (+/- {np.std(res['GCN (2 layers)']):.1%})")

# ---------------------------------------------------------------------------
# 4. Node embeddings from random walks (DeepWalk without the neural network)
# ---------------------------------------------------------------------------
print("\n4. Node embeddings")


def random_walks(A, walks_per_node=10, length=40, seed=0):
    rng = np.random.default_rng(seed)
    As = sparse.csr_matrix(A)
    ip, ix = As.indptr, As.indices
    d = np.diff(ip)
    cur = np.repeat(np.arange(len(A)), walks_per_node)
    W = [cur]
    for _ in range(length - 1):
        cur = ix[ip[cur] + (rng.random(len(cur)) * d[cur]).astype(int)]
        W.append(cur)
    return np.array(W).T


def walk_embedding(A, dim=16, window=5, seed=0):
    """Count how often nodes appear near each other on random walks, turn the
    counts into positive pointwise mutual information (PPMI), and factorise.
    Levy & Goldberg (2014) and Qiu et al. (2018) show this is what skip-gram /
    DeepWalk is implicitly doing."""
    W = random_walks(A, seed=seed)
    n = len(A)
    C = np.zeros((n, n))
    for k in range(1, window + 1):
        np.add.at(C, (W[:, :-k].ravel(), W[:, k:].ravel()), 1)
    C = C + C.T
    tot = C.sum()
    pmi = np.log(np.maximum(C * tot, 1e-12) / np.outer(C.sum(1), C.sum(0)))
    ppmi = np.maximum(pmi, 0)
    U, s, _ = np.linalg.svd(ppmi)
    return U[:, :dim] * np.sqrt(s[:dim])


t_walks = random_walks(A)
show("walks: count x length", f"{t_walks.shape[0]} x {t_walks.shape[1]}")
show("share of walk steps that stay within a type", f"{(y[t_walks[:, 1:]] == y[t_walks[:, :-1]]).mean():.1%}")
Zw = walk_embedding(A)
Lap = sparse.csgraph.laplacian(sparse.csr_matrix(A), normed=True)
vals, vecs = eigsh(Lap.astype(float), k=17, which="SM", tol=1e-8, v0=np.ones(n))
Zs = vecs[:, np.argsort(vals)][:, 1:17]  # drop the trivial first eigenvector
show("embedding shapes (walk / spectral)", f"{Zw.shape} / {Zs.shape}")
nz = np.linalg.norm(Zw, axis=1) + 1e-12
cos = Zw @ Zw.T / np.outer(nz, nz)
same = y[:, None] == y[None]
off = ~np.eye(n, dtype=bool)
show("mean cosine: same-type pairs vs different", f"{cos[same & off].mean():.2f} vs {cos[~same].mean():.2f}")
for lab in splits(n, K_LAB, REPS):
    res["walk embedding"].append(acc(logreg(Zw, y, lab), y, ~lab))
    res["walk embedding + features"].append(acc(logreg(np.column_stack([Zw, X]), y, lab), y, ~lab))
    res["spectral embedding"].append(acc(logreg(Zs, y, lab), y, ~lab))
for k in ["walk embedding", "spectral embedding", "walk embedding + features"]:
    show(k, f"{np.mean(res[k]):.1%}")

# ---------------------------------------------------------------------------
# 5. Link prediction: which streets are missing from the map?
# ---------------------------------------------------------------------------
print("\n5. Link prediction")
rng = np.random.default_rng(5)
perm = rng.permutation(len(E))
n_test = len(E) // 10
test_pos = E[perm[:n_test]]
train_E = E[perm[n_test:]]
A_tr = np.zeros_like(A)
A_tr[train_E[:, 0], train_E[:, 1]] = A_tr[train_E[:, 1], train_E[:, 0]] = 1
neg = []
while len(neg) < n_test:
    a, b = rng.integers(0, n, 2)
    if a != b and A[a, b] == 0:
        neg.append((a, b))
test_neg = np.array(neg)
pairs = np.vstack([test_pos, test_neg])
truth = np.r_[np.ones(n_test), np.zeros(n_test)]
show("held-out streets / non-streets", f"{n_test} / {n_test}")


def scores(Atr, pairs, Z=None):
    cn = (Atr[pairs[:, 0]] * Atr[pairs[:, 1]]).sum(1)
    w = 1 / np.log(np.maximum(Atr.sum(1), 2))
    aa = (Atr[pairs[:, 0]] * Atr[pairs[:, 1]] * w).sum(1)
    out = {"common neighbours": cn, "Adamic-Adar": aa}
    if Z is not None:
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        out["walk-embedding cosine"] = (Zn[pairs[:, 0]] * Zn[pairs[:, 1]]).sum(1)
    return out


d_xy = -np.sqrt(((xy[pairs[:, 0]] - xy[pairs[:, 1]]) ** 2).sum(1))
show("AUC: straight-line distance (cheating-free baseline)", f"{roc_auc_score(truth, d_xy):.3f}")
Zw_tr = walk_embedding(A_tr)
for k, s in scores(A_tr, pairs, Zw_tr).items():
    show(f"AUC: {k} (trained without test edges)", f"{roc_auc_score(truth, s):.3f}")
leak = scores(A, pairs, walk_embedding(A))
show("AUC: common neighbours, LEAKY full graph", f"{roc_auc_score(truth, leak['common neighbours']):.3f}")
show("AUC: walk-embedding cosine, LEAKY full graph", f"{roc_auc_score(truth, leak['walk-embedding cosine']):.3f}")
far = np.sqrt(((xy[test_neg[:, 0]] - xy[test_neg[:, 1]]) ** 2).sum(1))
show("random non-street pairs: median distance (km-ish)", f"{np.median(far):.1f}  vs streets < 0.42")
# harder negatives: pairs two hops apart in the training graph, which do look like streets
A2 = (A_tr @ A_tr > 0) & (A == 0) & ~np.eye(n, dtype=bool)
cand = np.array(np.nonzero(np.triu(A2))).T
hard = cand[rng.choice(len(cand), n_test, replace=False)]
pairs_h = np.vstack([test_pos, hard])
d_h = -np.sqrt(((xy[pairs_h[:, 0]] - xy[pairs_h[:, 1]]) ** 2).sum(1))
show("AUC with hard (2-hop) negatives: distance", f"{roc_auc_score(truth, d_h):.3f}")
for k, s in scores(A_tr, pairs_h, Zw_tr).items():
    show(f"AUC with hard (2-hop) negatives: {k}", f"{roc_auc_score(truth, s):.3f}")

# ---------------------------------------------------------------------------
# 6. When neighbours are unlike you: a trip graph
# ---------------------------------------------------------------------------
print("\n6. Heterophily: the trip graph")
# Edges now mean "lots of rides between these two stations". Commuters ride
# from residential stations to offices, so most edges join DIFFERENT types.
rng = np.random.default_rng(6)
pref = np.array([[0.05, 1.0, 0.25], [1.0, 0.05, 0.4], [0.25, 0.4, 0.05]])
P_edge = pref[y[:, None], y[None]] * (14.0 / n)
U = rng.random((n, n))
At = np.triu(U < P_edge, 1).astype(float)
At = At + At.T
lonely = At.sum(1) == 0
for i in np.where(lonely)[0]:
    j = rng.choice(np.where(y != y[i])[0])
    At[i, j] = At[j, i] = 1
Et = np.array(np.nonzero(np.triu(At))).T
show("trip graph: edges / homophily", f"{len(Et)} / {(y[Et[:, 0]] == y[Et[:, 1]]).mean():.1%}")
St = At / np.sqrt(At.sum(1))[:, None] / np.sqrt(At.sum(1))[None]
Ah_t = sym_norm(At)
Dt = At / At.sum(1, keepdims=True)
r = {k: [] for k in ["features only", "label propagation", "SGC (Ahat X, mixed)", "own + neighbour-mean (kept apart)", "own + neighbours' label shares"]}
for lab in splits(n, K_LAB, REPS):
    un = ~lab
    Y0 = np.zeros((n, 3))
    Y0[lab, y[lab]] = 1
    r["features only"].append(acc(logreg(X, y, lab), y, un))
    r["label propagation"].append(acc(label_prop(St, Y0).argmax(1), y, un))
    r["SGC (Ahat X, mixed)"].append(acc(logreg(Ah_t @ Xs, y, lab), y, un))
    r["own + neighbour-mean (kept apart)"].append(acc(logreg(np.column_stack([Xs, Dt @ Xs]), y, lab), y, un))
    r["own + neighbours' label shares"].append(acc(logreg(np.column_stack([Xs, Dt @ Xs, Dt @ Y0]), y, lab), y, un))
for k, v in r.items():
    show(k, f"{np.mean(v):.1%}")

# ---------------------------------------------------------------------------
# Summary table and figure
# ---------------------------------------------------------------------------
print("\nSummary (street graph, 60 labels, mean of 20 splits)")
order = ["majority", "features only", "1-hop vote", "label propagation", "features + LP scores",
         "SGC (A^2 X)", "GCN (2 layers)", "spectral embedding", "walk embedding", "walk embedding + features"]
for k in order:
    show(k, f"{np.mean(res[k]):.1%}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
cols = np.array(["tab:blue", "tab:orange", "tab:green"])
for a, b in E:
    ax[0].plot(xy[[a, b], 0], xy[[a, b], 1], color="lightgrey", lw=0.4, zorder=0)
lab0 = next(splits(n, K_LAB, 1))
ax[0].scatter(xy[:, 0], xy[:, 1], s=10, c=cols[y], zorder=1)
ax[0].scatter(xy[lab0, 0], xy[lab0, 1], s=50, facecolors="none", edgecolors="k", lw=1, zorder=2, label="surveyed")
ax[0].set_title("Stations by true type (circled = surveyed)")
ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_aspect("equal")
bigc = comp == np.bincount(comp).argmax()
Zb = Zw[bigc] - Zw[bigc].mean(0)
pc = Zb @ np.linalg.svd(Zb, full_matrices=False)[2][:2].T
ax[1].scatter(pc[:, 0], pc[:, 1], s=8, c=cols[y[bigc]], alpha=0.6)
ax[1].set_title("Walk embedding, main component (2-D PCA)")
ks = order[1:]
ax[2].barh(range(len(ks)), [np.mean(res[k]) for k in ks], color="tab:purple")
ax[2].set_yticks(range(len(ks)))
ax[2].set_yticklabels(ks, fontsize=8)
ax[2].axvline(np.mean(res["features only"]), color="grey", ls="--", lw=1)
ax[2].set_xlim(0.4, 1.0)
ax[2].set_title("Accuracy on unlabelled stations")
fig.tight_layout()
fig.savefig(OUT / "graph_models.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/graph_models.png")
