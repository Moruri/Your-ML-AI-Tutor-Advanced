"""
Lesson 016 - Collaborative filtering and matrix factorisation

Same city, same Explore tab as lesson 015, but a few months later: the app
now asks riders for a star rating after each visit, so there are enough
stars to learn from. Today we predict the stars a rider would give places
they haven't rated, first from neighbours and then by factorising the
rider-place matrix, and we look hard at the riders and places with no
history at all (cold start).

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/factors.png`, next to
this script (ignored by git), and runs in under a minute.
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


def rating_world(n_users=1200, n_items=240, seed=16):
    """Lesson 015's world, with busier riders and a prompt to rate every visit.
    Returns (user, item, day, stars) for every rating, plus the hidden truth."""
    rng = np.random.default_rng(seed)
    K, Z = len(GENRES), 6
    main = rng.integers(0, K, n_items)
    g = 0.25 * rng.random((n_items, K))
    g[np.arange(n_items), main] += 1.0
    quality = rng.normal(0, 0.5, n_items)
    izone = rng.integers(0, Z, n_items)
    vis = rng.lognormal(0, 0.8, n_items)
    taste = rng.gamma(0.6, 1.0, (n_users, K))
    taste = 1.6 * taste / taste.sum(1, keepdims=True)
    uzone = rng.integers(0, Z, n_users)
    dist = np.abs(uzone[:, None] - izone[None, :])
    util = taste @ g.T + quality[None, :] - 0.35 * dist + rng.normal(0, 0.3, (n_users, n_items))
    n_visits = np.clip(rng.lognormal(3.0, 0.6, n_users).astype(int), 3, 120)
    rows = []
    for u in range(n_users):
        w0 = np.exp(1.2 * util[u]) * vis
        items = rng.choice(n_items, size=min(n_visits[u], n_items), replace=False, p=w0 / w0.sum())
        days = np.sort(rng.integers(0, 365, len(items)))
        rows += [(u, i, d) for i, d in zip(items, days)]
    V = np.array(rows)
    z = (util[V[:, 0], V[:, 1]] - util.mean()) / util.std()
    generous = rng.normal(0, 0.5, n_users)  # some riders hand out stars, some never give a 5
    stars = np.clip(np.round(2.6 + 1.1 * z + generous[V[:, 0]] + rng.normal(0, 0.5, len(V))), 1, 5)
    rated = rng.random(len(V)) < 0.55 + 0.1 * np.abs(stars - 3)
    return dict(V=V[rated], stars=stars[rated], util=util, g=g, main=main, izone=izone, taste=taste,
                n_users=n_users, n_items=n_items)


W = rating_world()
V, S, U, I = W["V"], W["stars"], W["n_users"], W["n_items"]
uu, ii, dd = V[:, 0], V[:, 1], V[:, 2]

# ---------------------------------------------------------------------------
# 1. The ratings matrix
# ---------------------------------------------------------------------------
print("\n1. The rider-place matrix")
show("ratings / riders / places", f"{len(S)} / {U} / {I}")
show("matrix filled in", f"{len(S) / (U * I):.1%}")
show("ratings per rider: median / max", f"{np.median(np.bincount(uu, minlength=U)):.0f} / {np.bincount(uu).max()}")
show("star counts 1..5", " ".join(str(int((S == k).sum())) for k in range(1, 6)))

# Hold out 300 riders completely for the cold-start section. For everyone else:
# train on days 0-269, validate (pick settings) on 270-299, test on 300-364.
rng = np.random.default_rng(0)
cold_users = rng.choice(U, 300, replace=False)
warm = ~np.isin(uu, cold_users)
tr = warm & (dd < 270)
va = warm & (dd >= 270) & (dd < 300)
te = warm & (dd >= 300)
fit_tr = tr | va  # refit on train + validation once settings are chosen
show("train / validation / test ratings", f"{tr.sum()} / {va.sum()} / {te.sum()}")


def rmse(pred, mask):
    return np.sqrt(np.mean((np.clip(pred, 1, 5) - S[mask]) ** 2))


# ---------------------------------------------------------------------------
# 2. Baselines: the mean, and biases
# ---------------------------------------------------------------------------


def fit_biases(mask, lam=5.0, iters=10):
    """r ~ mu + b_user + b_item, ridge-shrunk, by alternating averages."""
    mu = S[mask].mean()
    bu, bi = np.zeros(U), np.zeros(I)
    u, i, r = uu[mask], ii[mask], S[mask]
    for _ in range(iters):
        bi = np.bincount(i, r - mu - bu[u], I) / (np.bincount(i, minlength=I) + lam)
        bu = np.bincount(u, r - mu - bi[i], U) / (np.bincount(u, minlength=U) + lam)
    return mu, bu, bi


print("\n2. Baselines (test RMSE in stars)")
mu = S[fit_tr].mean()
show("global mean", f"{rmse(np.full(te.sum(), mu), te):.3f}")
n_i = np.bincount(ii[fit_tr], minlength=I)
item_mean = np.bincount(ii[fit_tr], S[fit_tr], I) / np.maximum(n_i, 1)
show("raw place average", f"{rmse(np.where(n_i[ii[te]] > 0, item_mean[ii[te]], mu), te):.3f}")
mu, bu, bi = fit_biases(fit_tr)
base_pred = lambda m: mu + bu[uu[m]] + bi[ii[m]]
show("mean + rider bias + place bias (shrunk)", f"{rmse(base_pred(te), te):.3f}")

# ---------------------------------------------------------------------------
# 3. Neighbourhood CF: places like the ones you rated
# ---------------------------------------------------------------------------


def item_knn(mask, k=20, shrink=10):
    """Item-item similarity on bias-removed ratings, shrunk towards 0 when few
    riders rated both places. Predict from the k most similar places you rated."""
    m0, b_u, b_i = fit_biases(mask)
    R = np.zeros((U, I))
    M = np.zeros((U, I), bool)
    R[uu[mask], ii[mask]] = S[mask] - m0 - b_u[uu[mask]] - b_i[ii[mask]]
    M[uu[mask], ii[mask]] = True
    num = R.T @ R
    norm = np.sqrt((R ** 2).T @ M) * np.sqrt((R ** 2).T @ M).T
    co = M.T.astype(float) @ M
    sim = num / (norm + 1e-9) * co / (co + shrink)
    np.fill_diagonal(sim, 0)

    def predict(m):
        out = np.empty(m.sum())
        for n, (u, i) in enumerate(zip(uu[m], ii[m])):
            js = np.where(M[u])[0]
            s = sim[i, js]
            top = np.argsort(-s)[:k]
            s, js = s[top], js[top]
            keep = s > 0
            adj = (s[keep] @ R[u, js[keep]]) / (s[keep].sum() + 1e-9) if keep.any() else 0.0
            out[n] = m0 + b_u[u] + b_i[i] + adj
        return out

    return predict, sim


print("\n3. Item-item neighbours")
for k in (5, 20, 50):
    p, _ = item_knn(tr, k=k)
    show(f"k = {k:>2} neighbours: validation RMSE", f"{rmse(p(va), va):.3f}")
knn, SIM = item_knn(fit_tr, k=20)
show("k = 20, refit, test RMSE", f"{rmse(knn(te), te):.3f}")
probe = int(np.argmax(np.bincount(ii[fit_tr], minlength=I) * (W["main"] == 1)))
nb = np.argsort(-SIM[probe])[:5]
show(f"place {probe} ({GENRES[W['main'][probe]]}): 5 nearest", ", ".join(f"{j} ({GENRES[W['main'][j]]})" for j in nb))

# ---------------------------------------------------------------------------
# 4. Matrix factorisation by alternating least squares
# ---------------------------------------------------------------------------


def als(mask, k=4, lam=10.0, iters=12, seed=0):
    """r ~ mu + b_u + b_i + p_u . q_i. Fix the places, solve one small ridge
    regression per rider for (p_u, b_u); fix the riders, do the same per place."""
    g = np.random.default_rng(seed)
    mu0 = S[mask].mean()
    P, Q = 0.1 * g.standard_normal((U, k)), 0.1 * g.standard_normal((I, k))
    b_u, b_i = np.zeros(U), np.zeros(I)
    u, i, r = uu[mask], ii[mask], S[mask] - mu0
    by_u = [np.where(u == x)[0] for x in range(U)]
    by_i = [np.where(i == x)[0] for x in range(I)]
    reg = lam * np.eye(k + 1)
    for _ in range(iters):
        for x, rows in enumerate(by_u):
            if len(rows):
                A = np.column_stack([Q[i[rows]], np.ones(len(rows))])
                sol = np.linalg.solve(A.T @ A + reg, A.T @ (r[rows] - b_i[i[rows]]))
                P[x], b_u[x] = sol[:k], sol[k]
        for x, rows in enumerate(by_i):
            if len(rows):
                A = np.column_stack([P[u[rows]], np.ones(len(rows))])
                sol = np.linalg.solve(A.T @ A + reg, A.T @ (r[rows] - b_u[u[rows]]))
                Q[x], b_i[x] = sol[:k], sol[k]
    return dict(mu=mu0, P=P, Q=Q, bu=b_u, bi=b_i, lam=lam, k=k)


def mf_pred(m, mask):
    return m["mu"] + m["bu"][uu[mask]] + m["bi"][ii[mask]] + np.sum(m["P"][uu[mask]] * m["Q"][ii[mask]], 1)


print("\n4. Matrix factorisation (ALS): validation RMSE")
grid = {}
for k in (1, 2, 4, 8, 16):
    for lam in (1.0, 3.0, 10.0, 30.0):
        grid[(k, lam)] = rmse(mf_pred(als(tr, k, lam), va), va)
    show(f"k = {k:>2}, lambda 1 / 3 / 10 / 30", " / ".join(f"{grid[(k, l)]:.3f}" for l in (1.0, 3.0, 10.0, 30.0)))
best = min(grid, key=grid.get)
MF = als(fit_tr, *best)
show(f"chosen k = {best[0]}, lambda = {best[1]:.0f}: test RMSE", f"{rmse(mf_pred(MF, te), te):.3f}")

# The tempting shortcut: fill the gaps with 0 stars and take a truncated SVD.
R0 = np.zeros((U, I))
R0[uu[fit_tr], ii[fit_tr]] = S[fit_tr]
Uu, sv, Vt = np.linalg.svd(R0, full_matrices=False)
R4 = (Uu[:, :4] * sv[:4]) @ Vt[:4]
show("trap: zero-filled SVD, k = 4: test RMSE", f"{rmse(R4[uu[te], ii[te]], te):.3f}")
show("  ... its mean prediction on test pairs", f"{R4[uu[te], ii[te]].mean():.2f} stars (truth {S[te].mean():.2f})")

# ---------------------------------------------------------------------------
# 5. What did the factors learn?
# ---------------------------------------------------------------------------
print("\n5. Inside the factors")
MF8 = als(fit_tr, 8, 3.0)
for c, name in enumerate(GENRES):
    r2 = []
    for m in (MF, MF8):
        Xq = np.column_stack([m["Q"], np.ones(I)])
        res = W["g"][:, c] - Xq @ np.linalg.lstsq(Xq, W["g"][:, c], rcond=None)[0]
        r2.append(1 - res.var() / W["g"][:, c].var())
    show(f"R^2 of '{name}' from factors, k = {MF['k']} / k = 8", f"{r2[0]:.2f} / {r2[1]:.2f}")
show("corr(place bias, true average enjoyment)", f"{np.corrcoef(MF['bi'], W['util'].mean(0))[0, 1]:.2f}")

# Ranking check: for each test rider, rank the places they rated in the test
# window by predicted stars. NDCG@5 with relevance = stars - 1 (lesson 015's metric, graded).


def ndcg_within(pred, mask, k=5):
    out = []
    for u in np.unique(uu[mask]):
        rows = np.where(mask & (uu == u))[0]
        if len(rows) < 3:
            continue
        rel = S[rows] - 1
        p = pred[np.searchsorted(np.where(mask)[0], rows)]
        disc = 1 / np.log2(np.arange(2, min(k, len(rows)) + 2))
        got = rel[np.argsort(-p, kind="stable")][:k] @ disc
        ideal = np.sort(rel)[::-1][:k] @ disc
        out.append(got / ideal if ideal > 0 else 1.0)
    return np.mean(out), len(out)


g0 = np.random.default_rng(1)
for name, pred in [("random order", g0.random(te.sum())), ("place bias only", bi[ii[te]]),
                   ("item-item neighbours", knn(te)), ("matrix factorisation", mf_pred(MF, te))]:
    v, n = ndcg_within(pred, te)
    show(f"NDCG@5 within a rider's test places: {name}", f"{v:.3f}")
show("  (riders with 3+ test ratings)", n)

# ---------------------------------------------------------------------------
# 6. Cold start
# ---------------------------------------------------------------------------
# New riders: keep the place factors, and fold each new rider in with the same
# ridge solve ALS uses, from their first n ratings. Score their other ratings.
print("\n6. Cold start: new riders, first n ratings (RMSE on the rest)")
cold = np.isin(uu, cold_users)
cold_curve = []
for n in (0, 1, 2, 3, 5, 10):
    e_mf, e_b, cnt = [], [], 0
    for u in cold_users:
        rows = np.where(cold & (uu == u))[0]
        rows = rows[np.argsort(dd[rows], kind="stable")]
        if len(rows) < 15:
            continue
        cnt += 1
        known, rest = rows[:n], rows[10:]  # always score the same tail: rating 11 onwards
        A = np.column_stack([MF["Q"][ii[known]], np.ones(n)])
        sol = np.linalg.solve(A.T @ A + MF["lam"] * np.eye(MF["k"] + 1), A.T @ (S[known] - MF["mu"] - MF["bi"][ii[known]]))
        pred = MF["mu"] + MF["bi"][ii[rest]] + sol[-1] + MF["Q"][ii[rest]] @ sol[:-1]
        e_mf.append((np.clip(pred, 1, 5) - S[rest]) ** 2)
        b = (S[known] - MF["mu"] - MF["bi"][ii[known]]).sum() / (n + 5)
        e_b.append((np.clip(MF["mu"] + MF["bi"][ii[rest]] + b, 1, 5) - S[rest]) ** 2)
    r_mf, r_b = np.sqrt(np.concatenate(e_mf).mean()), np.sqrt(np.concatenate(e_b).mean())
    cold_curve.append((n, r_mf, r_b))
    show(f"n = {n:>2}: fold-in MF / biases only", f"{r_mf:.3f} / {r_b:.3f}")
show("  (new riders with 15+ ratings)", cnt)

# New places: no ratings at all, but the owner tagged a genre and gave an
# address when signing up. Learn a map from those tags to place factors and
# biases on the known places, then apply it to the new one.
print("\n6b. Cold start: new places, genre + zone tags only (leave-one-out over places)")
tags = np.column_stack([np.eye(4)[W["main"]], np.eye(6)[W["izone"]]])
e_tag, e_mean = [], []
for i0 in range(I):
    keep = np.arange(I) != i0
    B = np.linalg.lstsq(tags[keep], np.column_stack([MF["Q"], MF["bi"]])[keep], rcond=None)[0]
    q_new, b_new = (tags[i0] @ B)[:-1], (tags[i0] @ B)[-1]
    rows = np.where(te & (ii == i0))[0]
    pred = MF["mu"] + MF["bu"][uu[rows]] + b_new + MF["P"][uu[rows]] @ q_new
    e_tag.append((np.clip(pred, 1, 5) - S[rows]) ** 2)
    e_mean.append((np.clip(MF["mu"] + MF["bu"][uu[rows]], 1, 5) - S[rows]) ** 2)
show("rider bias only (place unknown)", f"{np.sqrt(np.concatenate(e_mean).mean()):.3f}")
show("genre + zone tags -> place factors", f"{np.sqrt(np.concatenate(e_tag).mean()):.3f}")
show("the same places once rated (full MF)", f"{rmse(mf_pred(MF, te), te):.3f}")

# ---------------------------------------------------------------------------
# Figure: place factors coloured by genre, and the cold-start curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
Qc = MF["Q"] - MF["Q"].mean(0)
pc = np.linalg.svd(Qc, full_matrices=False)[2][:2]
xy = Qc @ pc.T
for c, name in enumerate(GENRES):
    m = W["main"] == c
    ax[0].scatter(xy[m, 0], xy[m, 1], s=12, label=name)
ax[0].set_title(f"Place factors (k = {MF['k']}), first two directions")
ax[0].legend(fontsize=8)
cc = np.array(cold_curve)
ax[1].plot(cc[:, 0], cc[:, 1], "o-", label="fold-in MF")
ax[1].plot(cc[:, 0], cc[:, 2], "s--", label="biases only")
ax[1].set_xlabel("ratings known for the new rider")
ax[1].set_ylabel("RMSE on later ratings")
ax[1].set_title("Cold start: how fast a new rider warms up")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "factors.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/factors.png")
