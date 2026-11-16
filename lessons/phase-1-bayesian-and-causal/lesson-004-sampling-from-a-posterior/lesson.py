"""
Lesson 004 - Sampling from a posterior

Lesson 003 found the posterior on paper because the prior and likelihood
fitted together. Most real models don't. Change the prior a little, or add a
second unknown, and there's no formula left, and a grid with many unknowns is
far too big. The way out is to draw samples from the posterior instead of
writing it down.

Today we write a Metropolis-Hastings sampler from scratch in about fifteen
lines, test it on an answer we already know, tune it, check whether it has
mixed, and then use it on two models with no closed form.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/chains.png`, next to
this script (ignored by git), and takes about ten seconds.
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
rng = np.random.default_rng(4)


def show(label, value):
    print(f"  {label:<46} -> {value}")


# ---------------------------------------------------------------------------
# 1. The sampler
# ---------------------------------------------------------------------------
print("\n1. A Metropolis-Hastings sampler")


def metropolis(log_post, start, step, n_steps, rng):
    """Random-walk Metropolis. Returns the chain and the acceptance rate.

    log_post only needs to be right up to a constant: the constant cancels in
    the ratio, which is exactly why sampling works when we can't normalise.
    """
    x = np.atleast_1d(np.asarray(start, dtype=float))
    lp = log_post(x)
    chain = np.empty((n_steps, x.size))
    accepted = 0
    for i in range(n_steps):
        proposal = x + step * rng.standard_normal(x.size)
        lp_new = log_post(proposal)
        if np.log(rng.random()) < lp_new - lp:  # accept with prob min(1, ratio)
            x, lp = proposal, lp_new
            accepted += 1
        chain[i] = x
    return chain, accepted / n_steps


# Station A from lessons 002-003: Beta(2, 15) prior, 3 late out of 10.
late, n = 3, 10


def log_post_beta(x):
    t = x[0]
    if not 0 < t < 1:
        return -np.inf  # outside the allowed range: always rejected
    return (2 - 1 + late) * np.log(t) + (15 - 1 + n - late) * np.log(1 - t)


chain, acc = metropolis(log_post_beta, 0.5, 0.1, 20_000, rng)
show("first five values", np.round(chain[:5, 0], 3).tolist())
show("acceptance rate (step 0.1)", f"{acc:.2f}")

# ---------------------------------------------------------------------------
# 2. Check it against an answer we know: Beta(5, 22)
# ---------------------------------------------------------------------------
print("\n2. Check against Beta(5, 22)")

kept = chain[1000:, 0]  # drop the warm-up while it walks in from 0.5
lo, hi = np.quantile(kept, [0.025, 0.975])
show("sampler: mean, 95% interval", f"{kept.mean():.3f}  [{lo:.3f}, {hi:.3f}]")
lo, hi = stats.beta.ppf([0.025, 0.975], 5, 22)
show("exact:   mean, 95% interval", f"{5 / 27:.3f}  [{lo:.3f}, {hi:.3f}]")
show("sampler: P(rate > 0.25)", f"{(kept > 0.25).mean():.3f}")

# ---------------------------------------------------------------------------
# 3. Step size and effective sample size
# ---------------------------------------------------------------------------
print("\n3. Step size")


def ess(x):
    """Effective sample size: how many independent draws the chain is worth."""
    x = x - x.mean()
    m = len(x)
    acf = np.correlate(x, x, mode="full")[m - 1 :] / (x.var() * m)
    s = 0.0
    for k in range(1, m):
        if acf[k] < 0.05:  # stop once the correlation has died away
            break
        s += acf[k]
    return m / (1 + 2 * s)


traces = {}
for step in [0.005, 0.1, 1.5]:
    c, a = metropolis(log_post_beta, 0.5, step, 20_000, rng)
    traces[step] = c[:, 0]
    show(f"step {step:<5}: acceptance, ESS of 19,000", f"{a:.2f}, {ess(c[1000:, 0]):,.0f}")

# ---------------------------------------------------------------------------
# 4. Has it mixed? Several chains and R-hat
# ---------------------------------------------------------------------------
print("\n4. Has it mixed?")


def r_hat(chains):
    """Split R-hat: compares variance between and within half-chains."""
    halves = np.concatenate([np.split(c, 2) for c in chains])
    m, k = halves.shape
    within = halves.var(axis=1, ddof=1).mean()
    between = k * halves.mean(axis=1).var(ddof=1)
    return np.sqrt(((k - 1) / k * within + between / k) / within)


starts = [0.02, 0.3, 0.6, 0.95]
for steps, step in [(400, 0.005), (20_000, 0.1)]:
    cs = np.array([metropolis(log_post_beta, s, step, steps, rng)[0][:, 0] for s in starts])
    show(f"4 chains, step {step}, {steps:,} steps: R-hat", f"{r_hat(cs[:, steps // 10 :]):.3f}")
    show("  chain means", np.round(cs[:, steps // 10 :].mean(axis=1), 3).tolist())

# ---------------------------------------------------------------------------
# 5. A prior with no conjugate partner
# ---------------------------------------------------------------------------
print("\n5. A prior with no closed form")

# The network team describes stations on the log-odds scale: log-odds of a
# late return ~ Normal(-2.0, 0.6). That's a sensible prior, but it isn't a
# Beta, so lesson 003's shortcut doesn't apply. Sample on the log-odds scale.


def log_post_logit(x):
    z = x[0]
    p = 1 / (1 + np.exp(-z))
    return stats.norm.logpdf(z, -2.0, 0.6) + late * np.log(p) + (n - late) * np.log(1 - p)


cs = np.array([metropolis(log_post_logit, s, 0.6, 20_000, rng)[0][:, 0] for s in [-4, -2, 0, 2]])
show("R-hat", f"{r_hat(cs[:, 2000:]):.3f}")
rates = 1 / (1 + np.exp(-cs[:, 2000:].ravel()))
lo, hi = np.quantile(rates, [0.025, 0.975])
show("sampler: mean rate, 95% interval", f"{rates.mean():.3f}  [{lo:.3f}, {hi:.3f}]")
show("sampler: P(rate > 0.25)", f"{(rates > 0.25).mean():.3f}")
# One unknown, so we can still check on a grid.
z = np.linspace(-8, 4, 4001)
w = np.exp([log_post_logit([zi]) for zi in z])
w /= w.sum()
p = 1 / (1 + np.exp(-z))
show("grid check: mean rate, P(rate > 0.25)", f"{(w * p).sum():.3f}, {w[p > 0.25].sum():.3f}")

# ---------------------------------------------------------------------------
# 6. Two unknowns: does rain make returns late?
# ---------------------------------------------------------------------------
print("\n6. Two unknowns: the rain effect")

# 80 rides at Station A over a month, 30 of them on rainy days (simulated).
rain = np.r_[np.ones(30), np.zeros(50)]
true_p = 1 / (1 + np.exp(-(-2.0 + 0.9 * rain)))
is_late = (rng.random(80) < true_p).astype(float)
show("late on rainy / dry days", f"{is_late[rain == 1].sum():.0f}/30, {is_late[rain == 0].sum():.0f}/50")


def log_post_rain(x):
    alpha, beta = x
    eta = alpha + beta * rain
    loglik = (is_late * eta - np.logaddexp(0, eta)).sum()  # Bernoulli-logit
    return loglik + stats.norm.logpdf(alpha, -2, 1) + stats.norm.logpdf(beta, 0, 1)


cs = np.array(
    [metropolis(log_post_rain, s, 0.35, 20_000, rng)[0] for s in [(-4, -2), (0, 2), (-2, 0), (-1, -1)]]
)
kept = cs[:, 2000:]
show("R-hat for alpha, beta", f"{r_hat(kept[:, :, 0]):.3f}, {r_hat(kept[:, :, 1]):.3f}")
beta_draws = kept[:, :, 1].ravel()
show("rain effect beta: mean, 95% interval",
     "{:.2f}  [{:.2f}, {:.2f}]".format(beta_draws.mean(), *np.quantile(beta_draws, [0.025, 0.975])))
show("P(rain makes lateness more likely)", f"{(beta_draws > 0).mean():.3f}")
show("odds ratio exp(beta): median", f"{np.exp(np.median(beta_draws)):.2f}")
show("ESS for beta, chain 1", f"{ess(kept[0, :, 1]):,.0f}")

# ---------------------------------------------------------------------------
# Figure: trace plots for three step sizes, and the rain posterior
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, (step, tr) in zip(axes.ravel()[:3], traces.items()):
    ax.plot(tr[:3000], lw=0.6)
    ax.set_title(f"step {step}")
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("iteration")
ax = axes[1, 1]
ax.plot(kept[0, ::5, 0], kept[0, ::5, 1], ".", ms=1.5, alpha=0.4)
ax.set_xlabel("alpha (dry-day log-odds)")
ax.set_ylabel("beta (rain effect)")
ax.set_title("rain model, chain 1")
fig.tight_layout()
fig.savefig(OUT / "chains.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/chains.png")
