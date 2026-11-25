"""
Lesson 017 - Content, implicit feedback and ranking losses

Same city, same Explore tab. This time we throw the stars away and learn
from what riders *do*: which places they ride to, and how often they go
back. We fit two models built for that kind of data (weighted implicit ALS
and BPR, a pairwise ranking loss), add the places' genre and zone tags so
brand-new places aren't invisible, and judge everything on the list a rider
would actually see.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/ranking.png`, next to
this script (ignored by git), and runs in well under a minute.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def show(label, value):
    print(f"  {label:<46} -> {value}")


GENRES = ["food", "nature", "culture", "nightlife"]
Z = 6  # neighbourhoods along the main road


def click_world(n_users=1200, n_items=240, seed=17):
    """A year of rides to places, with repeat visits. Some places open late in
    the year. Nothing is rated: the log is just (rider, place, day)."""
    rng = np.random.default_rng(seed)
    K = len(GENRES)
    main = rng.integers(0, K, n_items)
    g = 0.25 * rng.random((n_items, K))
    g[np.arange(n_items), main] += 1.0
    quality = rng.normal(0, 0.5, n_items)
    izone = rng.integers(0, Z, n_items)
    vis = rng.lognormal(0, 0.8, n_items)
    opens = np.zeros(n_items, int)
    late = rng.choice(n_items, 30, replace=False)
    opens[late] = rng.integers(285, 330, 30)  # 30 places open around the end of training
    taste = rng.gamma(0.6, 1.0, (n_users, K))
    taste = 1.6 * taste / taste.sum(1, keepdims=True)
    uzone = rng.integers(0, Z, n_users)
    dist = np.abs(uzone[:, None] - izone[None, :])
    util = taste @ g.T + quality[None, :] - 0.35 * dist + rng.normal(0, 0.3, (n_users, n_items))
    n_rides = np.clip(rng.lognormal(2.9, 0.6, n_users).astype(int), 4, 150)
    rows = []
    for u in range(n_users):
        days = np.sort(rng.integers(0, 365, n_rides[u]))
        w = np.exp(2.0 * util[u]) * vis
        for d in days:
            ww = w * (opens <= d) * np.where((opens > 0) & (d - opens < 45), 2.0, 1.0)
            rows.append((u, rng.choice(n_items, p=ww / ww.sum()), d))
    V = np.array(rows)
    return dict(V=V, util=util, main=main, izone=izone, opens=opens, vis=vis,
                n_users=n_users, n_items=n_items)


W = click_world()
V, U, I = W["V"], W["n_users"], W["n_items"]
uu, ii, dd = V[:, 0], V[:, 1], V[:, 2]

# Content features for every place: one-hot genre, one-hot zone, and a bias column.
TAGS = np.zeros((I, len(GENRES) + Z + 1))
TAGS[np.arange(I), W["main"]] = 1
TAGS[np.arange(I), len(GENRES) + W["izone"]] = 1
TAGS[:, -1] = 1


def counts(mask):
    C = np.zeros((U, I))
    np.add.at(C, (uu[mask], ii[mask]), 1)
    return C


# ---------------------------------------------------------------------------
# 1. Clicks, not stars
# ---------------------------------------------------------------------------
print("\n1. What riders did (no ratings at all)")
C_all = counts(np.ones(len(V), bool))
show("riders / places / rides", f"{U} / {I} / {len(V)}")
show("distinct rider-place pairs", f"{int((C_all > 0).sum())} ({(C_all > 0).mean():.1%} of the matrix)")
show("rides that were repeat visits", f"{1 - (C_all > 0).sum() / len(V):.0%}")
show("rides per pair: median / max", f"{np.median(C_all[C_all > 0]):.0f} / {int(C_all.max())}")
lo, hi = np.percentile(W["util"], [25, 75])
pairs = C_all > 0
show("mean rides when enjoyment is top / bottom quarter",
     f"{C_all[pairs & (W['util'] > hi)].mean():.2f} / {C_all[pairs & (W['util'] < lo)].mean():.2f}")
show("never-visited pairs in the riders' top quarter", f"{(~pairs & (W['util'] > hi)).sum() / (W['util'] > hi).sum():.0%}")

# Splits by time. Train days 0-269, validate 270-299 (to tune), refit on 0-299, test 300-364.
# The question each time: which *new* places (not yet visited) will this rider go to?


def setup(cut, end):
    past = dd < cut
    fut = (dd >= cut) & (dd < end)
    C = counts(past)
    seen = C > 0
    T = counts(fut) > 0
    T &= ~seen
    users = np.where(seen.any(1) & T.any(1))[0]
    return C, seen, T, users


def evaluate(scores, seen, T, users, k=10, items=None):
    """Recall@k and NDCG@k over each rider's not-yet-visited places."""
    S = scores[users].astype(float).copy()
    S[seen[users]] = -np.inf
    if items is not None:  # restrict the candidate set
        S[:, ~items] = -np.inf
    Tu = T[users] if items is None else T[users] & items[None, :]
    keep = Tu.any(1)
    S, Tu = S[keep], Tu[keep]
    top = np.argsort(-S, 1)[:, :k]
    hit = np.take_along_axis(Tu, top, 1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = np.array([disc[:min(n, k)].sum() for n in Tu.sum(1)])
    return dict(recall=(hit.sum(1) / Tu.sum(1)).mean(), ndcg=((hit * disc).sum(1) / idcg).mean(), n=int(keep.sum()))


def fmt(r):
    return f"R@10 {r['recall']:.3f} | NDCG@10 {r['ndcg']:.3f}"


Ctr, seen_tr, Tva, u_va = setup(270, 300)
Cfit, seen_fit, Tte, u_te = setup(300, 365)
show("validation / test riders with a new place to find", f"{len(u_va)} / {len(u_te)}")

# ---------------------------------------------------------------------------
# 2. Baselines, and the trap of fitting only what you observed
# ---------------------------------------------------------------------------
print("\n2. Baselines (test)")
rng = np.random.default_rng(0)
res = {}
res["random"] = evaluate(rng.random((U, I)), seen_fit, Tte, u_te)
pop = (Cfit > 0).sum(0)
res["most popular"] = evaluate(np.tile(pop, (U, 1)), seen_fit, Tte, u_te)
for name in ("random", "most popular"):
    show(name, fmt(res[name]))


def als_explicit_observed(C, k=8, lam=1.0, iters=8, seed=0):
    """Lesson 016's recipe, applied blindly: fit only the observed cells.
    With clicks every observed cell is a 1, so there's nothing to learn."""
    r = np.random.default_rng(seed)
    P, Q = 0.1 * r.normal(size=(U, k)), 0.1 * r.normal(size=(I, k))
    M = C > 0
    for _ in range(iters):
        for u in range(U):
            js = M[u]
            P[u] = np.linalg.solve(Q[js].T @ Q[js] + lam * np.eye(k), Q[js].T @ np.ones(js.sum()))
        for i in range(I):
            us = M[:, i]
            Q[i] = np.linalg.solve(P[us].T @ P[us] + lam * np.eye(k), P[us].T @ np.ones(us.sum()))
    return P @ Q.T


S_obs = als_explicit_observed(Cfit)
res["MF on observed cells only"] = evaluate(S_obs, seen_fit, Tte, u_te)
show("MF on observed cells only (lesson 016 style)", fmt(res["MF on observed cells only"]))
show("  spread of its scores (std / mean)", f"{S_obs.std():.3f} / {S_obs.mean():.3f}")

# ---------------------------------------------------------------------------
# 3. Weighted implicit ALS: every cell counts, some count more
# ---------------------------------------------------------------------------


def implicit_als(C, k=16, lam=10.0, alpha=10.0, iters=8, seed=0):
    """Hu, Koren & Volinsky (2008). Preference p = 1 if visited else 0, for
    EVERY cell. Confidence c = 1 + alpha * log(1 + rides). Weighted ridge
    regressions, alternating between riders and places."""
    r = np.random.default_rng(seed)
    P, Q = 0.1 * r.normal(size=(U, k)), 0.1 * r.normal(size=(I, k))
    Pref = (C > 0).astype(float)
    Conf = 1 + alpha * np.log1p(C)
    E = lam * np.eye(k)

    def half(X, Y, Pm, Cm):
        YtY = Y.T @ Y
        for a in range(X.shape[0]):
            nz = Pm[a] > 0
            Yn, cn = Y[nz], Cm[a, nz]
            A = YtY + (Yn.T * (cn - 1)) @ Yn + E
            X[a] = np.linalg.solve(A, (Yn.T * cn).sum(1))

    for _ in range(iters):
        half(P, Q, Pref, Conf)
        half(Q, P, Pref.T, Conf.T)
    return P, Q


print("\n3. Implicit ALS: validation NDCG@10")
best = None
for k in (8, 16, 32):
    row = []
    for lam in (3.0, 30.0):
        for alpha in (3.0, 15.0):
            P, Q = implicit_als(Ctr, k, lam, alpha)
            r = evaluate(P @ Q.T, seen_tr, Tva, u_va)
            row.append(f"{r['ndcg']:.3f}")
            if best is None or r["ndcg"] > best[0]:
                best = (r["ndcg"], k, lam, alpha)
    show(f"k = {k:>2}, (lam, alpha) = (3,3) (3,15) (30,3) (30,15)", " / ".join(row))
_, K_ALS, LAM, ALPHA = best
P_als, Q_als = implicit_als(Cfit, K_ALS, LAM, ALPHA)
res["implicit ALS"] = evaluate(P_als @ Q_als.T, seen_fit, Tte, u_te)
show(f"chosen k = {K_ALS}, lam = {LAM:g}, alpha = {ALPHA:g}: test", fmt(res["implicit ALS"]))
P1, Q1 = implicit_als(Cfit, K_ALS, LAM, 0.0)
show("same, alpha = 0 (a visit weighs no more than a blank)", fmt(evaluate(P1 @ Q1.T, seen_fit, Tte, u_te)))

# ---------------------------------------------------------------------------
# 4. A ranking loss: BPR, with and without content
# ---------------------------------------------------------------------------


def train_bpr(C, k=16, epochs=40, lr=0.05, reg=0.01, use_tags=False, loss="bpr", seed=0, batch=256):
    """Bayesian Personalised Ranking (Rendle et al., 2009): for a rider u, a
    place i they visited and a place j they didn't, push score(u,i) above
    score(u,j). score(u,i) = p_u . (q_i + tags_i @ E) + b_i.
    loss="pointwise" instead treats i as label 1 and j as label 0 (logistic)."""
    r = np.random.default_rng(seed)
    P = 0.1 * r.normal(size=(U, k))
    Q = 0.1 * r.normal(size=(I, k))
    E = 0.1 * r.normal(size=(TAGS.shape[1], k)) if use_tags else np.zeros((TAGS.shape[1], k))
    b = np.zeros(I)
    pu, pi = np.nonzero(C > 0)
    M = C > 0
    for _ in range(epochs):
        order = r.permutation(len(pu))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            u, i = pu[idx], pi[idx]
            j = r.integers(0, I, len(idx))
            ok = ~M[u, j]
            u, i, j = u[ok], i[ok], j[ok]
            vi, vj = Q[i] + TAGS[i] @ E, Q[j] + TAGS[j] @ E
            si, sj = (P[u] * vi).sum(1) + b[i], (P[u] * vj).sum(1) + b[j]
            if loss == "bpr":
                gi = 1 / (1 + np.exp(si - sj))  # d log sigmoid(si - sj) / d si
                gj = -gi
            else:
                gi = 1 / (1 + np.exp(si))  # d log sigmoid(si) / d si
                gj = -1 / (1 + np.exp(-sj))  # d log sigmoid(-sj) / d sj
            dP = gi[:, None] * vi + gj[:, None] * vj - reg * P[u]
            dQi = gi[:, None] * P[u] - reg * Q[i]
            dQj = gj[:, None] * P[u] - reg * Q[j]
            np.add.at(P, u, lr * dP)
            np.add.at(Q, i, lr * dQi)
            np.add.at(Q, j, lr * dQj)
            np.add.at(b, i, lr * (gi - reg * b[i]))
            np.add.at(b, j, lr * (gj - reg * b[j]))
            if use_tags:
                g_e = TAGS[i].T @ (gi[:, None] * P[u]) + TAGS[j].T @ (gj[:, None] * P[u])
                E += lr * (g_e / len(u) * 8 - reg * E)
    return dict(P=P, Q=Q, E=E, b=b)


def bpr_scores(m, new=None):
    Q = m["Q"].copy()
    b = m["b"].copy()
    if new is not None:  # places with no training rides: no learned vector, content only
        Q[new] = 0
        b[new] = np.median(m["b"])
    return m["P"] @ (Q + TAGS @ m["E"]).T + b


print("\n4. Ranking losses (test)")
bpr = train_bpr(Cfit)
res["BPR"] = evaluate(bpr_scores(bpr), seen_fit, Tte, u_te)
show("BPR (pairwise ranking loss)", fmt(res["BPR"]))
pw = train_bpr(Cfit, loss="pointwise")
res["pointwise"] = evaluate(bpr_scores(pw), seen_fit, Tte, u_te)
show("same model, pointwise logistic loss", fmt(res["pointwise"]))
hyb = train_bpr(Cfit, use_tags=True)
res["BPR + tags"] = evaluate(bpr_scores(hyb), seen_fit, Tte, u_te)
show("BPR + genre/zone tags (hybrid)", fmt(res["BPR + tags"]))
oracle = evaluate(W["util"] + np.log(W["vis"]) / 2, seen_fit, Tte, u_te)
show("hidden truth: enjoyment x visibility (oracle)", fmt(oracle))

# ---------------------------------------------------------------------------
# 5. New places: content to the rescue
# ---------------------------------------------------------------------------
print("\n5. Places that opened after the training window")
new = (Cfit > 0).sum(0) == 0
show("places with no rides before day 300", f"{int(new.sum())}")
show("share of test discoveries that are new places", f"{Tte[:, new].sum() / Tte.sum():.0%}")
for name, S in [("random order", rng.random((U, I))),
                ("BPR (new places only ever seen as negatives)", bpr_scores(bpr)),
                ("hybrid, content only for new places", bpr_scores(hyb, new))]:
    r = evaluate(S, seen_fit, Tte, u_te, k=5, items=new)
    show(name, f"R@5 {r['recall']:.3f} | NDCG@5 {r['ndcg']:.3f} | riders {r['n']}")

# ---------------------------------------------------------------------------
# Figure: test NDCG@10 per model, and how BPR and ALS ranks agree
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
names = list(res)
ax[0].barh(names, [res[n]["ndcg"] for n in names])
ax[0].axvline(oracle["ndcg"], color="k", ls="--", lw=1, label="oracle")
ax[0].set_xlabel("test NDCG@10 (places the rider has not yet visited)")
ax[0].legend(fontsize=8)
ax[0].set_title("Learning from rides, not stars")
pop_rank = np.argsort(np.argsort(-pop))
u0 = u_te[0]
ex = np.argsort(-np.where(seen_fit[u0], -np.inf, (P_als @ Q_als.T)[u0]))[:10]  # unvisited places only
eb = np.argsort(-np.where(seen_fit[u0], -np.inf, bpr_scores(hyb)[u0]))[:10]
ax[1].scatter(pop_rank, (Cfit > 0).sum(0), s=8, c="lightgrey", label="all places")
ax[1].scatter(pop_rank[ex], (Cfit > 0).sum(0)[ex], s=30, marker="o", label="ALS top 10, one rider")
ax[1].scatter(pop_rank[eb], (Cfit > 0).sum(0)[eb], s=30, marker="x", label="hybrid BPR top 10")
ax[1].set_xlabel("popularity rank")
ax[1].set_ylabel("riders who visited (train)")
ax[1].set_title("One rider's top 10 against the popularity curve")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ranking.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/ranking.png")
