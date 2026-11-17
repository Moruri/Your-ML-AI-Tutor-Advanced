"""
Lesson 005 - Hierarchical models and partial pooling

Twelve stations, wildly different amounts of data. Treat each one on its own
and the tiny stations swing to extremes. Lump them all together and the real
differences vanish. A hierarchical model sits in between: every station gets
its own rate, but those rates are drawn from a shared distribution that the
data also learns. Small stations borrow strength from the group; big stations
mostly speak for themselves.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/shrinkage.png`, next to
this script (ignored by git), and takes a few seconds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(5)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def expit(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. The data: twelve stations, very different sizes
# ---------------------------------------------------------------------------
print("\n1. Twelve stations")

names = list("ABCDEFGHIJKL")
rides = np.array([400, 250, 180, 120, 90, 60, 40, 25, 15, 8, 5, 3])
# Simulated truth: station log-odds ~ Normal(-1.8, 0.5). In real life we never
# see these; here we keep them so we can mark everyone's homework.
true_rate = expit(rng.normal(-1.8, 0.5, size=12))
late = rng.binomial(rides, true_rate)
show("rides per station", rides.tolist())
show("late returns", late.tolist())

# ---------------------------------------------------------------------------
# 2. No pooling: every station on its own
# ---------------------------------------------------------------------------
print("\n2. No pooling")

raw = late / rides
show("raw rates", np.round(raw, 2).tolist())
show("smallest three stations", f"{late[-3]}/{rides[-3]}, {late[-2]}/{rides[-2]}, {late[-1]}/{rides[-1]}")

# ---------------------------------------------------------------------------
# 3. Complete pooling: one rate for everyone
# ---------------------------------------------------------------------------
print("\n3. Complete pooling")

pooled = late.sum() / rides.sum()
show("one shared rate", f"{pooled:.3f}")

# ---------------------------------------------------------------------------
# 4. Partial pooling: a hierarchical model
# ---------------------------------------------------------------------------
print("\n4. Partial pooling")

# Model:
#   mu  ~ Normal(-2, 1)          the typical station's log-odds
#   tau ~ HalfNormal(1)          how much stations differ from each other
#   z_j ~ Normal(mu, tau)        station j's own log-odds
#   late_j ~ Binomial(rides_j, expit(z_j))
#
# Fourteen unknowns, but only two of them (mu, tau) are shared. Given mu and
# tau, each z_j is a one-dimensional problem, so we can do the whole thing on
# grids: integrate each z_j out on a fine grid, which leaves an exact
# posterior over (mu, tau) on a 2-D grid. No sampler needed, no tuning, and
# nothing to diagnose. (Lesson 004's sampler would also work; exercise 4.)

mu_g = np.linspace(-4, 0, 121)
tau_g = np.linspace(0.01, 2.0, 120)
z_g = np.linspace(-7, 3, 501)
dz = z_g[1] - z_g[0]

# log p(late_j | z) for every station and every z on the grid: (12, nz)
p = expit(z_g)
loglik_z = stats.binom.logpmf(late[:, None], rides[:, None], p[None, :])

# For each (mu, tau): log p(late_j | mu, tau) = log of integral over z_j.
M, T = np.meshgrid(mu_g, tau_g, indexing="ij")            # (nmu, ntau)
logprior_z = stats.norm.logpdf(z_g[None, None, :], M[..., None], T[..., None])  # (nmu, ntau, nz)
log_marg = np.zeros(M.shape)
for j in range(12):
    a = logprior_z + loglik_z[j][None, None, :]
    mx = a.max(axis=-1, keepdims=True)
    log_marg += (mx[..., 0] + np.log(np.exp(a - mx).sum(axis=-1) * dz))

log_hyper = log_marg + stats.norm.logpdf(M, -2, 1) + stats.halfnorm.logpdf(T, scale=1)
w = np.exp(log_hyper - log_hyper.max())
w /= w.sum()

show("posterior mean of mu (as a rate)", f"{(w * expit(M)).sum():.3f}")
tau_post = w.sum(axis=0)
tau_mean = (tau_post * tau_g).sum()
cdf = np.cumsum(tau_post)
tau_lo, tau_hi = tau_g[np.searchsorted(cdf, 0.025)], tau_g[np.searchsorted(cdf, 0.975)]
show("posterior of tau: mean, 95% interval", f"{tau_mean:.2f}  [{tau_lo:.2f}, {tau_hi:.2f}]")

# Each station's posterior mean rate: average over (mu, tau), and for each
# (mu, tau) over z_j. Collapse the hyper-grid to the cells that matter.
keep = w > 1e-6 * w.max()
wk = w[keep] / w[keep].sum()
lp_k = logprior_z[keep]                                     # (ncells, nz)
partial = np.empty(12)
partial_lo = np.empty(12)
partial_hi = np.empty(12)
for j in range(12):
    a = lp_k + loglik_z[j][None, :]
    a = np.exp(a - a.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)                       # p(z_j | mu, tau, data)
    post_z = (wk[:, None] * a).sum(axis=0)                  # p(z_j | data)
    partial[j] = (post_z * p).sum()
    c = np.cumsum(post_z)
    partial_lo[j], partial_hi[j] = p[np.searchsorted(c, 0.025)], p[np.searchsorted(c, 0.975)]

show("partially pooled rates", np.round(partial, 2).tolist())
for j in [0, 9, 10, 11]:
    show(f"  station {names[j]} ({late[j]}/{rides[j]}): raw -> pooled",
         f"{raw[j]:.2f} -> {partial[j]:.2f}  [{partial_lo[j]:.2f}, {partial_hi[j]:.2f}]")

# ---------------------------------------------------------------------------
# 5. Marking the homework against the truth
# ---------------------------------------------------------------------------
print("\n5. Which estimate was closest?")

for label, est in [("no pooling", raw), ("complete pooling", np.full(12, pooled)), ("partial pooling", partial)]:
    err = np.abs(est - true_rate)
    show(f"{label}: error (all / small 6)", f"{err.mean():.3f} / {err[6:].mean():.3f}")

# One simulation could be luck. Repeat the whole story 200 times with fresh
# stations and fresh rides, using a cheaper stand-in for partial pooling: a
# Beta prior fitted to the group by the method of moments (lesson 003), then
# a conjugate update per station. This is "empirical Bayes".


def eb_shrink(k, n):
    r = k / n
    m = k.sum() / n.sum()
    # variance of the true rates = observed spread minus binomial noise
    v = max(np.average((r - m) ** 2, weights=n) - m * (1 - m) / n.mean(), 1e-4)
    s = m * (1 - m) / v - 1                 # prior "pseudo-rides"
    return (k + m * s) / (n + s)


errs = np.zeros((200, 3))
for i in range(200):
    t = expit(rng.normal(-1.8, 0.5, size=12))
    k = rng.binomial(rides, t)
    for col, est in enumerate([k / rides, np.full(12, k.sum() / rides.sum()), eb_shrink(k, rides)]):
        errs[i, col] = np.abs(est - t).mean()
show("200 repeats: mean abs error (none/full/EB)", np.round(errs.mean(axis=0), 3).tolist())
show("EB beat no pooling in", f"{(errs[:, 2] < errs[:, 0]).mean():.0%} of repeats")

# ---------------------------------------------------------------------------
# 6. How much does each station shrink?
# ---------------------------------------------------------------------------
print("\n6. Shrinkage depends on data")

# On the log-odds scale, a station's estimate is roughly a weighted average of
# its own data and the group mean, with weight n_j / (n_j + kappa).
m = (w * expit(M)).sum()
kappa = 1 / (tau_mean**2 * m * (1 - m))
for j in [0, 4, 8, 11]:
    show(f"  station {names[j]} ({rides[j]} rides): weight on own data", f"{rides[j] / (rides[j] + kappa):.2f}")
show("rides needed to trust yourself half the time", f"{kappa:.0f}")

# ---------------------------------------------------------------------------
# Figure: raw rates shrinking towards the group
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
for j in range(12):
    ax.plot([0, 1], [raw[j], partial[j]], "-", color="grey", lw=0.8)
    ax.scatter([0, 1], [raw[j], partial[j]], s=12 + rides[j] / 2, color=["tab:red", "tab:blue"], zorder=3)
ax.axhline(pooled, ls="--", color="k", lw=0.8, label="complete pooling")
ax.set_xticks([0, 1], ["no pooling (raw)", "partial pooling"])
ax.set_xlim(-0.2, 1.2)
ax.set_ylabel("late-return rate")
ax.set_title("Small stations shrink most; dot size = rides")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "shrinkage.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/shrinkage.png")
