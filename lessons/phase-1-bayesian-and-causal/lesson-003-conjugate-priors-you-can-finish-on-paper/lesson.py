"""
Lesson 003 - Conjugate priors you can finish on paper

Lesson 002 updated a belief about Station A's late-return rate on a grid of
1,001 candidate rates. It worked, but it was brute force. For this model there
is a shortcut: if the prior is a Beta distribution, the posterior is another
Beta distribution, and you get it by adding two counts. No grid, no loop.

Today we do that update by hand, check it against the grid, and learn to read
the result as a plain statement about uncertainty.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/conjugate.png`, next to
this script (ignored by git), and takes a couple of seconds.
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
rng = np.random.default_rng(3)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def describe(name, a, b):
    """Mean and 95% equal-tailed interval of a Beta(a, b)."""
    lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
    show(f"{name}: mean, 95% interval", f"{a / (a + b):.3f}  [{lo:.3f}, {hi:.3f}]")


# ---------------------------------------------------------------------------
# 1. The Beta distribution, and the prior we already used
# ---------------------------------------------------------------------------
print("\n1. The Beta distribution")

# Beta(a, b) has density proportional to theta^(a-1) * (1-theta)^(b-1).
# Lesson 002's network prior was theta^1 (1-theta)^14, so it is Beta(2, 15),
# and the flat prior is Beta(1, 1).
a0, b0 = 2, 15
describe("network prior Beta(2, 15)", a0, b0)
describe("flat prior Beta(1, 1)", 1, 1)
show("network prior: mode (a-1)/(a+b-2)", f"{(a0 - 1) / (a0 + b0 - 2):.3f}")
show("network prior: P(rate > 0.25)", f"{stats.beta.sf(0.25, a0, b0):.3f}")

# ---------------------------------------------------------------------------
# 2. The update by hand: add the counts
# ---------------------------------------------------------------------------
print("\n2. The update by hand")

late, n = 3, 10
a1, b1 = a0 + late, b0 + (n - late)  # that's the whole update
show("Station A: 3 late, 7 on time", f"Beta({a0}+{late}, {b0}+{n - late}) = Beta({a1}, {b1})")
describe("posterior", a1, b1)

# Check it against lesson 002's grid.
theta = np.linspace(0, 1, 1001)
grid = theta ** (a0 - 1) * (1 - theta) ** (b0 - 1) * theta**late * (1 - theta) ** (n - late)
grid /= grid.sum()
show("grid posterior mean (lesson 002)", f"{(theta * grid).sum():.3f}")
exact = stats.beta.pdf(theta, a1, b1)
exact /= exact.sum()
show("max gap, grid vs Beta(5, 22)", f"{np.abs(grid - exact).max():.1e}")

# ---------------------------------------------------------------------------
# 3. Reading the posterior
# ---------------------------------------------------------------------------
print("\n3. Reading the posterior")

post = stats.beta(a1, b1)
show("mode (single most plausible rate)", f"{(a1 - 1) / (a1 + b1 - 2):.3f}")
show("standard deviation", f"{post.std():.3f}")
show("50% interval", "[{:.3f}, {:.3f}]".format(*post.ppf([0.25, 0.75])))
show("P(rate > 0.25)", f"{post.sf(0.25):.3f}")
show("P(rate < 0.10)", f"{post.cdf(0.10):.3f}")
show("rate we're 90% sure it's below", f"{post.ppf(0.90):.3f}")

# ---------------------------------------------------------------------------
# 4. The prior is pseudo-data
# ---------------------------------------------------------------------------
print("\n4. The prior is pseudo-data")

# Beta(2, 15) behaves like 17 rides we've already seen, 2 of them late.
# The posterior mean is a weighted average of the prior mean and the data.
prior_n = a0 + b0
w = n / (n + prior_n)
blend = w * (late / n) + (1 - w) * (a0 / prior_n)
show("prior worth this many rides", f"{prior_n}")
show("weight on the data, n / (n + a + b)", f"{w:.3f}")
show("blend of 0.300 and 0.118", f"{blend:.3f}")
for extra in [10, 100, 1000]:
    k = round(0.3 * extra)
    show(f"after {extra:>4} rides at 30% late: mean", f"{(a0 + k) / (prior_n + extra):.3f}")

# ---------------------------------------------------------------------------
# 5. Choosing a prior from what the network tells us
# ---------------------------------------------------------------------------
print("\n5. Choosing a prior from the network")

# Last quarter's late rates at 24 other stations (simulated, but realistic).
network_rates = rng.beta(2.4, 18, size=24)
m, v = network_rates.mean(), network_rates.var(ddof=1)
show("24 stations: mean, sd of late rate", f"{m:.3f}, {np.sqrt(v):.3f}")
# Method of moments: pick (a, b) so the Beta has that mean and variance.
strength = m * (1 - m) / v - 1
a_mm, b_mm = m * strength, (1 - m) * strength
show("matching Beta(a, b)", f"Beta({a_mm:.1f}, {b_mm:.1f}), worth {strength:.0f} rides")
describe("Station A with that prior", a_mm + late, b_mm + n - late)

# ---------------------------------------------------------------------------
# 6. Comparing two stations: a first taste of sampling
# ---------------------------------------------------------------------------
print("\n6. Is Station A worse than Station C?")

# Station C: 9 late out of 60. Same prior, separate updates.
aC, bC = a0 + 9, b0 + 51
describe("Station C, Beta(11, 66)", aC, bC)
draws_A = stats.beta.rvs(a1, b1, size=100_000, random_state=rng)
draws_C = stats.beta.rvs(aC, bC, size=100_000, random_state=rng)
diff = draws_A - draws_C
show("P(A worse than C), from 100k draws", f"{(diff > 0).mean():.3f}")
show("difference A - C: mean", f"{diff.mean():.3f}")
show("difference A - C: 95% interval", "[{:.3f}, {:.3f}]".format(*np.quantile(diff, [0.025, 0.975])))

# ---------------------------------------------------------------------------
# Figure: prior, posterior, and the grid checking it
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
xs = np.linspace(0, 0.8, 400)
ax.plot(xs, stats.beta.pdf(xs, a0, b0), label="prior Beta(2, 15)")
ax.plot(xs, stats.beta.pdf(xs, a1, b1), label="posterior Beta(5, 22)")
ax.plot(xs, stats.beta.pdf(xs, aC, bC), label="Station C Beta(11, 66)", ls="--")
ax.plot(theta[::20], grid[::20] * 1000, "k.", ms=3, label="lesson 002 grid (A)")
ax.axvline(0.25, color="grey", ls=":", lw=1)
ax.set_xlabel("late-return rate")
ax.set_ylabel("density")
ax.set_xlim(0, 0.6)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "conjugate.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/conjugate.png")
