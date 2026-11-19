"""
Lesson 009 - Quasi-experiments when you can't randomise

Lesson 008 ended on the hard case: the thing that drives treatment isn't in
your data, so no adjustment set can save you. Today we sidestep it with a
design instead of a model. The city painted protected bike lanes past some of
our stations. Nobody flipped a coin, and the stations that got lanes were not
typical. But we have weekly rides from before and after, at stations with and
without lanes. Difference-in-differences turns that into an estimate.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/did.png`, next to this
script (ignored by git), and runs in a couple of seconds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(5)


def show(label, value):
    print(f"  {label:<46} -> {value}")


# 60 stations, 26 weeks. Lanes open at the start of week 16 (weeks 0-15 are
# "before", 16-25 "after") next to 20 stations. The city picked busy stations
# on main roads, so treated stations start out busier: an unmeasured
# difference in level. Every station shares the same weekly ups and downs
# (weather, school holidays, a long rainy spell after week 16), and each
# station's own noise is sticky from week to week.
S, W, T0, N_TREAT = 60, 26, 16, 20
EFFECT = 6.0  # true extra rides per station per week, from the lanes
WEEKS = np.arange(W)


def world(trend_gap=0.0, effect=EFFECT):
    """Weekly rides, shape (S, W), and the treated flag, shape (S,).

    trend_gap adds extra growth (rides per week, per week) to treated stations
    only: what happens if the city put lanes where demand was already rising.
    """
    treated = np.zeros(S, bool)
    treated[:N_TREAT] = True
    level = rng.normal(80, 12, S) + 25 * treated  # busier main-road stations
    season = 8 * np.sin(2 * np.pi * WEEKS / 26) - 10 * (WEEKS >= T0) * (WEEKS < T0 + 6)
    noise = np.zeros((S, W))
    for w in range(W):  # AR(1): this week's luck carries into next week
        noise[:, w] = (0.6 * noise[:, w - 1] if w else 0) + rng.normal(0, 4, S)
    post = WEEKS >= T0
    y = (level[:, None] + season[None, :] + noise
         + trend_gap * treated[:, None] * WEEKS[None, :]
         + effect * treated[:, None] * post[None, :])
    return y, treated


def did_2x2(y, treated, t0=T0, start=0, stop=W):
    """Four means, two differences. Returns the estimate and the four means."""
    pre, post = slice(start, t0), slice(t0, stop)
    m = {
        ("T", "pre"): y[treated, pre].mean(), ("T", "post"): y[treated, post].mean(),
        ("C", "pre"): y[~treated, pre].mean(), ("C", "post"): y[~treated, post].mean(),
    }
    est = (m["T", "post"] - m["T", "pre"]) - (m["C", "post"] - m["C", "pre"])
    return est, m


# ---------------------------------------------------------------------------
# 1. Two tempting comparisons, both wrong
# ---------------------------------------------------------------------------
print("\n1. Two tempting comparisons")
y, treated = world()
est, m = did_2x2(y, treated)
show("mean weekly rides, lane stations: pre / post", f"{m['T', 'pre']:.1f} / {m['T', 'post']:.1f}")
show("mean weekly rides, other stations: pre / post", f"{m['C', 'pre']:.1f} / {m['C', 'post']:.1f}")
show("after only: lanes minus no lanes", f"{m['T', 'post'] - m['C', 'post']:+.1f}  (true {EFFECT:+.1f})")
show("lane stations only: after minus before", f"{m['T', 'post'] - m['T', 'pre']:+.1f}")

# ---------------------------------------------------------------------------
# 2. Difference-in-differences
# ---------------------------------------------------------------------------
print("\n2. Difference-in-differences")
show("change at lane stations", f"{m['T', 'post'] - m['T', 'pre']:+.2f}")
show("change at other stations", f"{m['C', 'post'] - m['C', 'pre']:+.2f}")
show("DiD estimate", f"{est:+.2f}  (true {EFFECT:+.1f})")

# ---------------------------------------------------------------------------
# 3. The same thing as a regression, and an honest standard error
# ---------------------------------------------------------------------------
print("\n3. DiD as a regression")


def ols(X, yv, groups=None):
    """OLS with plain or cluster-robust (by group) standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ yv
    e = yv - X @ b
    if groups is None:
        cov = e @ e / (len(yv) - X.shape[1]) * XtX_inv
    else:
        meat = np.zeros((X.shape[1], X.shape[1]))
        for g in np.unique(groups):
            s = X[groups == g].T @ e[groups == g]
            meat += np.outer(s, s)
        G = len(np.unique(groups))
        cov = G / (G - 1) * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(cov))


def did_regression(y, treated, groups=True):
    """rides ~ station dummies + week dummies + lanes_open. Returns (b, se)."""
    station = np.repeat(np.arange(S), W)
    week = np.tile(WEEKS, S)
    d = (treated[station] & (week >= T0)).astype(float)
    X = np.column_stack([d, np.eye(S)[station], np.eye(W)[week][:, 1:]])
    b, se = ols(X, y.ravel(), station if groups else None)
    return b[0], se[0]


b_plain, se_plain = did_regression(y, treated, groups=False)
b_cl, se_cl = did_regression(y, treated)
show("regression coefficient on lanes_open", f"{b_cl:+.2f}")
show("SE, plain OLS / clustered by station", f"{se_plain:.2f} / {se_cl:.2f}")
show("95% CI (clustered)", f"[{b_cl - 1.96 * se_cl:+.2f}, {b_cl + 1.96 * se_cl:+.2f}]")

# Is the plain SE really too small? Re-run the whole world 500 times.
reps = 500
cov_plain = cov_cl = 0
ests = []
for _ in range(reps):
    yr, tr = world()
    bp, sp = did_regression(yr, tr, groups=False)
    bc, sc = did_regression(yr, tr)
    ests.append(bc)
    cov_plain += abs(bp - EFFECT) <= 1.96 * sp
    cov_cl += abs(bc - EFFECT) <= 1.96 * sc
show(f"mean DiD estimate over {reps} worlds", f"{np.mean(ests):+.2f}")
show("95% CI covers the truth: plain / clustered", f"{cov_plain / reps:.1%} / {cov_cl / reps:.1%}")

# ---------------------------------------------------------------------------
# 4. Check the assumption you can check: event study and placebo
# ---------------------------------------------------------------------------
print("\n4. Parallel trends: event study and placebo")


def event_study(y, treated):
    """Treated-minus-control gap each week, relative to the last pre week."""
    gap = y[treated].mean(0) - y[~treated].mean(0)
    return gap - gap[T0 - 1]


def placebo(y, treated, fake_t0=8):
    """Pretend lanes opened at week 8 and use only pre-period data."""
    return did_2x2(y, treated, t0=fake_t0, start=0, stop=T0)[0]


es_good = event_study(y, treated)
show("pre-period gaps (weeks 0-14), mean |gap|", f"{np.abs(es_good[:T0 - 1]).mean():.2f}")
show("post-period gaps (weeks 16-25), mean", f"{es_good[T0:].mean():+.2f}")
show("placebo DiD, fake start at week 8", f"{placebo(y, treated):+.2f}")

# ---------------------------------------------------------------------------
# 5. When parallel trends fail
# ---------------------------------------------------------------------------
print("\n5. When the lanes went where demand was already growing")
y_bad, tr_bad = world(trend_gap=0.5)
est_bad, _ = did_2x2(y_bad, tr_bad)
show("DiD estimate (true effect still +6.0)", f"{est_bad:+.2f}")
es_bad = event_study(y_bad, tr_bad)
show("pre-period gap: week 0 vs week 14", f"{es_bad[0]:+.2f} vs {es_bad[T0 - 2]:+.2f}")
show("placebo DiD, fake start at week 8", f"{placebo(y_bad, tr_bad):+.2f}")
# One repair: fit the pre-period gap trend and project it forward. This only
# works if the trend would have continued in a straight line, which is itself
# an assumption the data can't confirm.
gap = y_bad[tr_bad].mean(0) - y_bad[~tr_bad].mean(0)
slope, icept = np.polyfit(WEEKS[:T0], gap[:T0], 1)
adj = (gap[T0:] - (icept + slope * WEEKS[T0:])).mean()
show("pre-trend slope of the gap (rides/week/week)", f"{slope:+.2f}")
show("DiD after projecting the pre-trend forward", f"{adj:+.2f}")

# A no-effect world with the same diverging trend: plain DiD 'finds' lanes work.
y_null, tr_null = world(trend_gap=0.5, effect=0.0)
show("no-effect world, plain DiD", f"{did_2x2(y_null, tr_null)[0]:+.2f}")

# ---------------------------------------------------------------------------
# Figure: the raw series and the two event studies
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(WEEKS, y[treated].mean(0), label="lane stations")
axes[0].plot(WEEKS, y[~treated].mean(0), label="other stations")
cf = y[~treated].mean(0) + (m["T", "pre"] - m["C", "pre"])
axes[0].plot(WEEKS[T0 - 1:], cf[T0 - 1:], "--", color="C0", alpha=0.6, label="lane stations, no lanes (DiD guess)")
axes[0].axvline(T0 - 0.5, color="grey", lw=0.8)
axes[0].set_xlabel("week")
axes[0].set_ylabel("mean weekly rides per station")
axes[0].set_title("Before and after, with a control")
axes[0].legend(fontsize=8)
axes[1].plot(WEEKS, es_good, "o-", ms=3, label="parallel trends hold")
axes[1].plot(WEEKS, es_bad, "s-", ms=3, label="lanes went to growing stations")
axes[1].axhline(0, color="grey", lw=0.8)
axes[1].axhline(EFFECT, color="k", ls=":", label="true effect")
axes[1].axvline(T0 - 0.5, color="grey", lw=0.8)
axes[1].set_xlabel("week")
axes[1].set_ylabel("gap vs week 15")
axes[1].set_title("Event study: look before you trust")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "did.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/did.png")
