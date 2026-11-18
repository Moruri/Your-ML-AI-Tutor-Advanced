"""
Lesson 007 - Randomised experiments and A/B tests

Lesson 006 showed that last month's reminder texts can't tell us whether texts
work, because operations chose who got them. The fix is to let a coin choose.
Today we design that experiment properly: decide the sample size before we
start, randomise, check the coin did its job, estimate the average treatment
effect with a confidence interval, and see the two classic ways good
experiments go bad (too small, and peeking).

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/experiments.png`, next
to this script (ignored by git), and runs in a few seconds.
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
rng = np.random.default_rng(7)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def expit(z):
    return 1 / (1 + np.exp(-z))


# The same rider world as lesson 006: past late rides (of 10), trip distance,
# a chance of being late without a text (p0) and with one (p1).
def riders(n):
    past_late = rng.binomial(10, rng.beta(1.2, 6, size=n))
    dist = rng.gamma(2.0, 1.5, size=n)
    z0 = -2.6 + 0.45 * past_late + 0.15 * dist
    effect = -1.6 * np.exp(-0.5 * ((past_late - 2) / 1.2) ** 2) * np.exp(-dist / 4)
    return past_late, dist, expit(z0), expit(z0 + effect)


# The true average effect of texting a typical rider, from a huge population.
_, _, P0, P1 = riders(500_000)
TRUE_ATE = (P1 - P0).mean()
BASE = P0.mean()

# ---------------------------------------------------------------------------
# 1. Design before data: how many riders do we need?
# ---------------------------------------------------------------------------
print("\n1. Sample size, decided in advance")


def n_per_arm(p, mde, alpha=0.05, power=0.8):
    """Riders per arm to detect a drop of `mde` from base rate `p`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil((z_a + z_b) ** 2 * 2 * p * (1 - p) / mde**2))


show("base late rate (no text)", f"{BASE:.3f}")
for mde in [0.05, 0.03, 0.02, 0.01]:
    show(f"  riders per arm to detect -{mde:.2f}", f"{n_per_arm(BASE, mde):,}")
N = n_per_arm(BASE, 0.03)
show("chosen design: riders per arm (MDE 0.03)", f"{N:,}")

# ---------------------------------------------------------------------------
# 2. Randomise, then check the coin
# ---------------------------------------------------------------------------
print("\n2. Randomise and check balance")

past_late, dist, p0, p1 = riders(2 * N)
t = rng.permutation(np.repeat([0, 1], N))  # exactly N in each arm, order random
y = (rng.random(2 * N) < np.where(t == 1, p1, p0)).astype(int)


def smd(x):
    """Standardised mean difference between arms (|SMD| < 0.1 is fine)."""
    a, b = x[t == 1], x[t == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


show("mean past_late: text / control", f"{past_late[t == 1].mean():.2f} / {past_late[t == 0].mean():.2f}")
show("SMD past_late, SMD dist", f"{smd(past_late):+.3f}, {smd(dist):+.3f}")

# ---------------------------------------------------------------------------
# 3. Estimate the effect with a confidence interval
# ---------------------------------------------------------------------------
print("\n3. The estimate")


def diff_ci(y, t):
    a, b = y[t == 1], y[t == 0]
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, se, d - 1.96 * se, d + 1.96 * se


d, se, lo, hi = diff_ci(y, t)
p_value = 2 * stats.norm.sf(abs(d / se))
show("late rate: text / control", f"{y[t == 1].mean():.3f} / {y[t == 0].mean():.3f}")
show("ATE estimate (text minus control)", f"{d:+.4f}  (true {TRUE_ATE:+.4f})")
show("standard error", f"{se:.4f}")
show("95% confidence interval", f"[{lo:+.4f}, {hi:+.4f}]")
show("p-value (two-sided z-test)", f"{p_value:.1e}")

# A check that leans on nothing but the randomisation itself: shuffle the
# labels many times and see how often chance alone gives a gap this big.
shuffled = np.array([rng.permutation(t) for _ in range(5000)])
null = (shuffled * y).sum(1) / N - ((1 - shuffled) * y).sum(1) / N
show("shuffles with a gap at least this big", f"{(np.abs(null) >= abs(d)).sum()} of 5,000")
show("largest gap chance produced", f"{np.abs(null).max():.4f}")

# ---------------------------------------------------------------------------
# 4. Same data, tighter interval: adjust for a pre-experiment covariate
# ---------------------------------------------------------------------------
print("\n4. Regression adjustment")


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    cov = resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov))


# past_late was recorded BEFORE the coin was flipped, so it can't be affected
# by the text. Centring it keeps the treatment coefficient an average effect.
X = np.column_stack([np.ones(2 * N), t, past_late - past_late.mean(), dist - dist.mean()])
b, ses = ols(X, y)
show("adjusted ATE estimate", f"{b[1]:+.4f}")
show("standard error: plain / adjusted", f"{se:.4f} / {ses[1]:.4f}")
show("like running the plain test with", f"{(se / ses[1]) ** 2:.2f}x the riders")

# ---------------------------------------------------------------------------
# 5. Run the experiment 2,000 times
# ---------------------------------------------------------------------------
print("\n5. What happens across many experiments")


def many(n, reps=2000, scale=1.0):
    """Repeat an n-per-arm experiment; return estimates, CI bounds and p-values."""
    _, _, q0, q1 = riders(2 * n * reps)
    q1 = q0 + scale * (q1 - q0)  # scale < 1 shrinks every rider's effect
    q0, q1 = q0.reshape(reps, 2 * n), q1.reshape(reps, 2 * n)
    a = (rng.random((reps, n)) < q1[:, :n]).astype(float)  # first n texted
    c = (rng.random((reps, n)) < q0[:, n:]).astype(float)  # rest control
    d = a.mean(1) - c.mean(1)
    se = np.sqrt(a.var(1, ddof=1) / n + c.var(1, ddof=1) / n)
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(np.abs(d / se))


d_big, lo_big, hi_big, p_big = many(N)
cover = ((lo_big <= TRUE_ATE) & (TRUE_ATE <= hi_big)).mean()
show(f"n={N:,}/arm: CI covers the truth", f"{cover:.1%}")
show(f"n={N:,}/arm: power (p < 0.05)", f"{(p_big < 0.05).mean():.1%}")
show(f"n={N:,}/arm: mean estimate when significant", f"{d_big[p_big < 0.05].mean():+.4f}")
# Power is promised at the MDE. Shrink every effect so the truth is about -0.03:
d_mde, _, _, p_mde = many(N, scale=0.03 / -TRUE_ATE)
show(f"n={N:,}/arm, true effect -0.03: power", f"{(p_mde < 0.05).mean():.1%}")
d_small, _, _, p_small = many(400)
show("n=400/arm: power (p < 0.05)", f"{(p_small < 0.05).mean():.1%}")
show("n=400/arm: mean estimate when significant", f"{d_small[p_small < 0.05].mean():+.4f}")
show("n=400/arm: exaggeration when significant", f"{d_small[p_small < 0.05].mean() / TRUE_ATE:.1f}x")

# ---------------------------------------------------------------------------
# 6. Peeking: check every day, stop when it looks good
# ---------------------------------------------------------------------------
print("\n6. Peeking at an A/A test")

# An A/A test: both arms get NOTHING, so any 'significant' result is false.
# 20 days, 150 new riders per arm per day, base late rate as above.
reps, days, per_day = 4000, 20, 150
a = rng.binomial(per_day, BASE, size=(reps, days)).cumsum(1)
c = rng.binomial(per_day, BASE, size=(reps, days)).cumsum(1)
n_so_far = per_day * np.arange(1, days + 1)
pa, pc = a / n_so_far, c / n_so_far
pool = (a + c) / (2 * n_so_far)
z = (pa - pc) / np.sqrt(pool * (1 - pool) * 2 / n_so_far)
sig = np.abs(z) > 1.96
looks = [1, 2, 5, 10, 20]
fpr = []
for k in looks:
    idx = np.linspace(days / k - 1, days - 1, k).round().astype(int)  # evenly spaced looks
    fpr.append(sig[:, idx].any(1).mean())
    show(f"  {k:>2} look(s): false positive rate", f"{fpr[-1]:.1%}")

# ---------------------------------------------------------------------------
# Figure: 50 confidence intervals, and the cost of peeking
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i in range(50):
    miss = not (lo_big[i] <= TRUE_ATE <= hi_big[i])
    axes[0].plot([lo_big[i], hi_big[i]], [i, i], color="crimson" if miss else "steelblue")
    axes[0].plot(d_big[i], i, "o", ms=3, color="crimson" if miss else "steelblue")
axes[0].axvline(TRUE_ATE, color="k", ls="--", label="true ATE")
axes[0].axvline(0, color="grey", lw=0.8)
axes[0].set_xlabel("effect of a text on late rate")
axes[0].set_ylabel("experiment")
axes[0].set_title(f"50 experiments, 95% CIs (n={N:,}/arm)")
axes[0].legend(loc="upper left")
axes[1].plot(looks, fpr, "o-")
axes[1].axhline(0.05, color="grey", ls="--", label="promised 5%")
axes[1].set_xlabel("number of looks at the data")
axes[1].set_ylabel("A/A tests declared 'significant'")
axes[1].set_title("Peeking inflates false positives")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "experiments.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/experiments.png")
