"""
Lesson 002 - Probability as belief

Lesson 001 ended with a question the bootstrap answers awkwardly: after only
ten rides, how late is Station A, really? Today we answer it the Bayesian way.
A belief about an unknown rate becomes a whole distribution, and Bayes' rule
moves that distribution when evidence arrives.

We do everything on a grid of 1,001 candidate rates with plain NumPy, so every
step is a multiplication you can print. Lesson 003 shows how to get the same
answers on paper.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/beliefs.png`, next to
this script (ignored by git), and takes a couple of seconds.
"""

from math import comb
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


# ---------------------------------------------------------------------------
# 1. Bayes' rule with counts: the fraud flag
# ---------------------------------------------------------------------------
print("\n1. Bayes' rule with counts")

# 1% of top-ups are fraud. The flag catches 95% of fraud and wrongly flags
# 5% of honest top-ups. A top-up is flagged. How likely is it to be fraud?
p_fraud, p_flag_if_fraud, p_flag_if_ok = 0.01, 0.95, 0.05
per_10k = 10_000
frauds = per_10k * p_fraud
flagged_fraud = frauds * p_flag_if_fraud
flagged_ok = (per_10k - frauds) * p_flag_if_ok
show("of 10,000 top-ups: fraud, flagged fraud", f"{frauds:.0f}, {flagged_fraud:.0f}")
show("honest but flagged", f"{flagged_ok:.0f}")
posterior = flagged_fraud / (flagged_fraud + flagged_ok)
show("P(fraud | flagged), counting", f"{posterior:.3f}")

evidence = p_flag_if_fraud * p_fraud + p_flag_if_ok * (1 - p_fraud)
show("P(fraud | flagged), Bayes' rule", f"{p_flag_if_fraud * p_fraud / evidence:.3f}")

# A second, independent check also says "fraud". Yesterday's posterior is
# today's prior.
p2 = p_flag_if_fraud * posterior
show("after a second independent flag", f"{p2 / (p2 + p_flag_if_ok * (1 - posterior)):.3f}")

# ---------------------------------------------------------------------------
# 2. A belief is a distribution over a grid
# ---------------------------------------------------------------------------
print("\n2. A belief is a distribution")

theta = np.linspace(0, 1, 1001)  # candidate late-return rates


def normalise(w):
    return w / w.sum()


def mean(p):
    return float((theta * p).sum())


def interval(p, mass=0.95):
    cdf = np.cumsum(p)
    lo = theta[np.searchsorted(cdf, (1 - mass) / 2)]
    hi = theta[np.searchsorted(cdf, 1 - (1 - mass) / 2)]
    return float(lo), float(hi)


def prob_above(p, cut):
    return float(p[theta > cut].sum())


# Prior 1: "no idea". Every rate equally plausible.
flat = normalise(np.ones_like(theta))
# Prior 2: what we know from the network. Most stations run near 10% late,
# rarely above 30%. Shape theta^1 (1-theta)^14 puts its peak at about 0.07.
network = normalise(theta**1 * (1 - theta) ** 14)

for name, prior in [("flat prior", flat), ("network prior", network)]:
    lo, hi = interval(prior)
    show(f"{name}: mean, 95% interval", f"{mean(prior):.3f}  [{lo:.3f}, {hi:.3f}]")
    show(f"{name}: P(rate > 0.25)", f"{prob_above(prior, 0.25):.3f}")
show("each prior sums to", f"{flat.sum():.3f}, {network.sum():.3f}")

# ---------------------------------------------------------------------------
# 3. Update with Station A's evidence: 3 late out of 10
# ---------------------------------------------------------------------------
print("\n3. Update with the evidence")

late, n = 3, 10
likelihood = comb(n, late) * theta**late * (1 - theta) ** (n - late)
show("likelihood peaks at rate", f"{theta[likelihood.argmax()]:.3f}")

posts = {}
for name, prior in [("flat", flat), ("network", network)]:
    post = normalise(prior * likelihood)  # Bayes' rule, all 1,001 at once
    posts[name] = post
    lo, hi = interval(post)
    show(f"{name} posterior: mean, 95% interval", f"{mean(post):.3f}  [{lo:.3f}, {hi:.3f}]")
    show(f"{name} posterior: P(rate > 0.25)", f"{prob_above(post, 0.25):.3f}")

# ---------------------------------------------------------------------------
# 4. One ride at a time gives the same answer
# ---------------------------------------------------------------------------
print("\n4. One ride at a time")

rides = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]  # 1 = returned late
belief = network.copy()
for i, r in enumerate(rides, 1):
    belief = normalise(belief * (theta if r else 1 - theta))
    if i in (1, 2, 5, 10):
        show(f"after ride {i:>2} ({sum(rides[:i])} late): mean", f"{mean(belief):.3f}")
gap = np.abs(belief - posts["network"]).max()
show("max gap vs all-at-once posterior", f"{gap:.1e}")

# ---------------------------------------------------------------------------
# 5. With lots of data, the prior stops mattering
# ---------------------------------------------------------------------------
print("\n5. Station B: 300 late out of 1,000")

like_b = theta**300 * (1 - theta) ** 700  # comb() cancels when we normalise
for name, prior in [("flat", flat), ("network", network)]:
    post = normalise(prior * like_b)
    lo, hi = interval(post)
    show(f"{name} posterior: mean, 95% interval", f"{mean(post):.3f}  [{lo:.3f}, {hi:.3f}]")

# ---------------------------------------------------------------------------
# 6. Predict the next rides
# ---------------------------------------------------------------------------
print("\n6. Predict what happens next")

post = posts["network"]
show("P(next ride at A is late)", f"{mean(post):.3f}")
# Chance of 3 or more late in the next 10, averaging over our uncertainty.
p_3plus = sum(
    (post * comb(10, k) * theta**k * (1 - theta) ** (10 - k)).sum() for k in range(3, 11)
)
plug_in = mean(post)
p_3plus_plug = sum(comb(10, k) * plug_in**k * (1 - plug_in) ** (10 - k) for k in range(3, 11))
show("P(3+ late in next 10), full posterior", f"{p_3plus:.3f}")
show("P(3+ late in next 10), plug in the mean", f"{p_3plus_plug:.3f}")

# ---------------------------------------------------------------------------
# Figure: prior, likelihood and posterior for Station A
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(theta, network, label="network prior")
ax.plot(theta, normalise(likelihood), label="likelihood (3 of 10), scaled")
ax.plot(theta, posts["network"], label="posterior")
ax.axvline(0.25, color="grey", ls=":", lw=1)
ax.set_xlabel("late-return rate at Station A")
ax.set_ylabel("probability on the grid")
ax.set_xlim(0, 0.8)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "beliefs.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/beliefs.png")
