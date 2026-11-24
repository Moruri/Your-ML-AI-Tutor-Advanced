"""
Lesson 015 - Recommender systems: the problem and the traps

The bike-share app is getting an "Explore" tab: a short list of places
(cafes, parks, markets, galleries, bars) each rider might like to cycle to
next. Today we don't build anything clever. We frame the job as
"score every place, show the top 10", build honest scoreboards for that
job, and walk into the traps that make recommenders look better on paper
than they are in the app.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/recs.png`, next to this
script (ignored by git), and runs in a few seconds.
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


def explore_world(n_users=1200, n_items=240, seed=15):
    """A year of riders visiting places. Everything a real app would log
    (who went where, when, and the odd star rating), plus the hidden truth
    (how much each rider would enjoy each place) that a real app never sees."""
    rng = np.random.default_rng(seed)
    K, Z = len(GENRES), 6  # taste dimensions, neighbourhoods
    # Places: one main genre, a dash of the others, a hidden quality, a zone,
    # a "visibility" (signage, Instagram, being on the main road) and an opening day.
    main = rng.integers(0, K, n_items)
    g = 0.25 * rng.random((n_items, K))
    g[np.arange(n_items), main] += 1.0
    quality = rng.normal(0, 0.5, n_items)
    izone = rng.integers(0, Z, n_items)
    vis = rng.lognormal(0, 0.8, n_items)
    opens = np.where(rng.random(n_items) < 0.15, rng.integers(120, 330, n_items), 0)  # 15% open mid-year
    # Riders: a taste vector (most people love one or two genres), a home zone, an activity level.
    taste = rng.gamma(0.6, 1.0, (n_users, K))
    taste = 1.6 * taste / taste.sum(1, keepdims=True)
    uzone = rng.integers(0, Z, n_users)
    dist = np.abs(uzone[:, None] - izone[None, :])  # zones lie along one long road
    util = taste @ g.T + quality[None, :] - 0.35 * dist + rng.normal(0, 0.3, (n_users, n_items))
    n_visits = np.clip(rng.lognormal(2.2, 0.7, n_users).astype(int), 2, 80)

    visits = []  # (user, item, day)
    for u in range(n_users):
        days = np.sort(rng.integers(0, 365, n_visits[u]))
        seen = set()
        for d in days:
            open_now = opens <= d
            novelty = np.where((d - opens < 60) & (opens > 0), 3.0, 1.0)  # new places get a buzz
            w = np.exp(2.0 * util[u]) * vis * novelty * open_now
            w[list(seen)] = 0
            i = rng.choice(n_items, p=w / w.sum())
            seen.add(i)
            visits.append((u, i, d))
    V = np.array(visits)
    # Stars: only some visits get rated, and people rate the extremes more often
    # (missing not at random). Stars follow how much this rider liked this place.
    u_, i_ = V[:, 0], V[:, 1]
    def to_stars(x, size):
        return np.clip(np.round(2.6 + 1.1 * (x - util.mean()) / util.std() + rng.normal(0, 0.5, size)), 1, 5)

    stars = to_stars(util[u_, i_], len(V))
    stars_all = to_stars(util, util.shape)  # what everyone would give everywhere: never observed
    p_rate = 0.12 + 0.12 * np.abs(stars - 3)
    rated = rng.random(len(V)) < p_rate
    return dict(V=V, stars=stars, stars_all=stars_all, rated=rated, util=util, g=g, main=main, quality=quality,
                vis=vis, opens=opens, uzone=uzone, izone=izone, n_users=n_users, n_items=n_items)


W = explore_world()
V, U, I = W["V"], W["n_users"], W["n_items"]

# ---------------------------------------------------------------------------
# 1. What the app has logged
# ---------------------------------------------------------------------------
print("\n1. A year of the Explore tab's raw material")
show("riders / places / visits", f"{U} / {I} / {len(V)}")
show("visits per rider: median / max", f"{np.median(np.bincount(V[:, 0])):.0f} / {np.bincount(V[:, 0]).max()}")
show("share of rider-place pairs ever visited", f"{len(V) / (U * I):.1%}")
cnt = np.bincount(V[:, 1], minlength=I)
top = np.sort(cnt)[::-1]
show("visits to the top 10% of places", f"{top[:I // 10].sum() / len(V):.0%}")
show("visits with a star rating", f"{W['rated'].sum()} ({W['rated'].mean():.0%})")

# ---------------------------------------------------------------------------
# 2. Accuracy is the wrong scoreboard
# ---------------------------------------------------------------------------
# Split by time: the past (days 0-299) is what you train on, the last 65 days
# are what the app has to predict. Test riders: those with history and a future.
CUT = 300
tr, te = V[V[:, 2] < CUT], V[V[:, 2] >= CUT]
users = np.intersect1d(np.unique(tr[:, 0]), np.unique(te[:, 0]))
seen_tr = np.zeros((U, I), bool)
seen_tr[tr[:, 0], tr[:, 1]] = True
truth = np.zeros((U, I), bool)
truth[te[:, 0], te[:, 1]] = True

print("\n2. Accuracy: 'nobody goes anywhere'")
cand = ~seen_tr[users]
acc = 1 - truth[users][cand].mean()
show("test riders / future visits to predict", f"{len(users)} / {truth[users].sum()}")
show("accuracy of predicting 'no visit' for every pair", f"{acc:.2%}")
show("useful places it recommends", "0")

# ---------------------------------------------------------------------------
# 3. Ranking metrics, by hand
# ---------------------------------------------------------------------------


def ranked(scores, seen):
    """Top items per row, never re-recommending something already visited."""
    s = np.where(seen, -np.inf, scores)
    return np.argsort(-s, axis=1, kind="stable")


def metrics(order, rel, k=10):
    """order: (n_users, n_items) ranking; rel: (n_users, n_items) bool truth."""
    topk = order[:, :k]
    hits = np.take_along_axis(rel, topk, axis=1)
    n_rel = rel.sum(1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    idcg = np.array([disc[:min(n, k)].sum() for n in n_rel])
    return dict(prec=hits.mean(1).mean(), rec=(hits.sum(1) / n_rel).mean(), hit=(hits.sum(1) > 0).mean(),
                ndcg=(dcg / idcg).mean(), cov=len(np.unique(topk)) / rel.shape[1])


print("\n3. One rider, one list of 5")
toy = np.array([True, False, True, False, False])  # hits at positions 1 and 3; rider made 3 future visits in total
disc = 1 / np.log2(np.arange(2, 7))
show("hits at positions", "1 and 3, out of 3 future visits")
show("precision@5 / recall@5", f"{toy.mean():.2f} / {toy.sum() / 3:.2f}")
show("DCG = 1/log2(2) + 1/log2(4)", f"{(toy * disc).sum():.3f}")
show("ideal DCG (3 hits at the top)", f"{disc[:3].sum():.3f}")
show("NDCG@5", f"{(toy * disc).sum() / disc[:3].sum():.3f}")
toy2 = np.array([False, False, True, False, True])
show("same 2 hits, at positions 3 and 5: NDCG@5", f"{(toy2 * disc).sum() / disc[:3].sum():.3f}")

# ---------------------------------------------------------------------------
# 4. Baselines on an honest, time-based split
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
pop = np.bincount(tr[:, 1], minlength=I).astype(float)
zone_pop = np.zeros((6, I))
np.add.at(zone_pop, (W["uzone"][tr[:, 0]], tr[:, 1]), 1)
# A taste profile from history: genre mix of the places you went, times popularity.
prof = np.zeros((U, len(GENRES)))
np.add.at(prof, tr[:, 0], W["g"][tr[:, 1]])
prof /= prof.sum(1, keepdims=True) + 1e-9

SCORERS = {
    "random": lambda us: rng.random((len(us), I)),
    "most popular": lambda us: np.tile(pop, (len(us), 1)),
    "popular in your zone": lambda us: zone_pop[W["uzone"][us]] + 0.1 * pop,
    "your genres x popularity": lambda us: (prof[us] @ W["g"].T) ** 3 * (zone_pop[W["uzone"][us]] + 0.1 * pop + 1),
    "true enjoyment (oracle)": lambda us: W["util"][us],
}


def evaluate(scorers, us, seen, rel):
    out = {}
    for name, f in scorers.items():
        out[name] = metrics(ranked(f(us), seen[us]), rel[us])
    return out


def report(res):
    for name, m in res.items():
        show(name, f"P@10 {m['prec']:.3f} | R@10 {m['rec']:.3f} | NDCG@10 {m['ndcg']:.3f} | hit {m['hit']:.0%} | cover {m['cov']:.0%}")


print("\n4. Time split (train before day 300, test after), top 10 per rider")
res = evaluate(SCORERS, users, seen_tr, truth)
report(res)
new_places = W["opens"] >= CUT - 60
show("future visits that go to places opened since day 240", f"{truth[users][:, new_places].sum() / truth[users].sum():.0%}")

# ---------------------------------------------------------------------------
# 5. Trap 1: the random split
# ---------------------------------------------------------------------------
# Hold out a random 18% of each rider's visits instead (about the same number
# as the time split's test set). Popularity is now counted partly from the future.
print("\n5. Trap 1: random split vs time split")
mask = rng.random(len(V)) < len(te) / len(V)
tr_r, te_r = V[~mask], V[mask]
users_r = np.intersect1d(np.unique(tr_r[:, 0]), np.unique(te_r[:, 0]))
seen_r = np.zeros((U, I), bool)
seen_r[tr_r[:, 0], tr_r[:, 1]] = True
truth_r = np.zeros((U, I), bool)
truth_r[te_r[:, 0], te_r[:, 1]] = True
pop_r = np.bincount(tr_r[:, 1], minlength=I).astype(float)
for name, f, us, s, t in [("most popular, time split", lambda us: np.tile(pop, (len(us), 1)), users, seen_tr, truth),
                          ("most popular, random split", lambda us: np.tile(pop_r, (len(us), 1)), users_r, seen_r, truth_r)]:
    m = metrics(ranked(f(us), s[us]), t[us])
    show(name, f"NDCG@10 {m['ndcg']:.3f} | hit {m['hit']:.0%}")

# ---------------------------------------------------------------------------
# 6. Trap 2: "sort by average rating"
# ---------------------------------------------------------------------------
print("\n6. Trap 2: the average star rating")
R = W["rated"]
ri, rs = V[R, 1], W["stars"][R]
n_r = np.bincount(ri, minlength=I)
s_r = np.bincount(ri, weights=rs, minlength=I)
mean_r = np.where(n_r > 0, s_r / np.maximum(n_r, 1), 0)
mu = rs.mean()
m_prior = 10  # pretend every place starts with 10 average ratings
shrunk = (s_r + m_prior * mu) / (n_r + m_prior)
true_avg = W["util"].mean(0)  # how much the average rider would enjoy it


def top_true_rank(score, k=10):
    """Mean true-enjoyment rank (1 = best of 240) of the k places a score puts first."""
    rank = np.empty(I, int)
    rank[np.argsort(-true_avg)] = np.arange(1, I + 1)
    t = np.argsort(-score)[:k]
    return rank[t].mean(), np.median(n_r[t])


show("mean star given / share of 4-5 stars", f"{mu:.2f} / {(rs >= 4).mean():.0%}")
show("if every rider rated every place (hidden)", f"{W['stars_all'].mean():.2f} / {(W['stars_all'] >= 4).mean():.0%}")
show("places with 1-3 ratings", f"{((n_r > 0) & (n_r <= 3)).sum()}")
for name, sc in [("top 10 by raw average", mean_r), ("top 10 by shrunk average (m=10)", shrunk), ("top 10 by visit count", cnt.astype(float))]:
    r, n = top_true_rank(sc)
    show(name, f"mean true rank {r:5.1f} of {I}, median ratings each {n:4.0f}")

# ---------------------------------------------------------------------------
# 7. Trap 3: the feedback loop
# ---------------------------------------------------------------------------
# The app goes live. Each week every rider sees 10 places and, sometimes, goes to one.
# Policy A shows the 10 most-visited places in your zone (from the app's own logs).
# Policy B does the same but swaps 3 slots for random picks from the zone's top 60.
print("\n7. Trap 3: twelve weeks of a live app")


def live(explore, weeks=12, seed=7):
    g = np.random.default_rng(seed)
    counts = zone_pop.copy()
    shown_ever = np.zeros(I, bool)
    joy = []
    for _ in range(weeks):
        week_joy = []
        for z in range(6):
            us = np.where(W["uzone"] == z)[0]
            order = np.argsort(-counts[z])
            in_zone = order[:60]
            for u in us:
                slate = list(order[:10 - explore])
                if explore:
                    rest = [i for i in in_zone if i not in slate]
                    slate += list(g.choice(rest, explore, replace=False))
                slate = np.array(slate)
                shown_ever[slate] = True
                p = 1 / (1 + np.exp(-(2 * W["util"][u, slate] - 3)))  # tempted by places you'd enjoy
                go = g.random(len(slate)) < p / 3
                if go.any():
                    i = slate[np.argmax(np.where(go, W["util"][u, slate], -np.inf))]
                    counts[z, i] += 1
                    week_joy.append(W["util"][u, i])
        joy.append(np.mean(week_joy))
    share = np.sort(counts.sum(0) - zone_pop.sum(0))[::-1]
    return shown_ever.mean(), share[:I // 10].sum() / share.sum(), np.mean(joy[:2]), np.mean(joy[-2:]), joy


loop = {}
for name, e in [("A: top 10 by your zone's visits", 0), ("B: top 7 + 3 random picks", 3)]:
    cov_, conc, j0, j1, joy = live(e)
    loop[name] = joy
    show(name, f"places ever shown {cov_:.0%}, app visits to top 10% {conc:.0%}, enjoyment wk 1-2 {j0:.2f} -> wk 11-12 {j1:.2f}")

# ---------------------------------------------------------------------------
# Figure: the long tail, and the scoreboard
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(np.arange(I), top, width=1.0, color="C0")
ax[0].set_xlabel("places, most visited first")
ax[0].set_ylabel("visits in the year")
ax[0].set_title("The long tail")
names = list(res)
ax[1].barh(names[::-1], [res[n]["ndcg"] for n in names[::-1]], color="C2")
ax[1].set_xlabel("NDCG@10 (time split)")
ax[1].set_title("Baselines on the honest split")
fig.tight_layout()
fig.savefig(OUT / "recs.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/recs.png")
