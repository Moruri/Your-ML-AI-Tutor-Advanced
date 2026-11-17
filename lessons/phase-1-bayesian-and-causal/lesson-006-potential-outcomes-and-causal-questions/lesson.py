"""
Lesson 006 - Potential outcomes and causal questions

"Who is most likely to return late?" is a prediction question. "Who should we
text a reminder?" is a decision question, and it's causal: it asks what would
happen to each rider WITH the text and WITHOUT it. Only one of those ever
happens, so the answer can't be read straight off the data.

Today we write the decision down as a contrast of potential outcomes, see why
a naive comparison misleads, and watch an excellent predictor make a poor
policy.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/uplift.png`, next to
this script (ignored by git), and runs in a second or two.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(6)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def expit(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. A world where we can see both outcomes
# ---------------------------------------------------------------------------
print("\n1. Both potential outcomes (simulation only)")

# 20,000 riders. Two things we know about each: how many of their last 10
# rides were late, and how far their usual trip is (km).
n = 20_000
past_late = rng.binomial(10, rng.beta(1.2, 6, size=n))
dist = rng.gamma(2.0, 1.5, size=n)

# Y0 = late WITHOUT a reminder, Y1 = late WITH one. We simulate both, which
# real life never lets us do.
z0 = -2.6 + 0.45 * past_late + 0.15 * dist
p0 = expit(z0)
# The reminder helps forgetful occasional riders most. Riders who are late
# almost every time, or who ride a long way, are late for reasons a text
# can't fix (long trips, shifts that run over), so it barely moves them.
effect_logodds = -1.6 * np.exp(-0.5 * ((past_late - 2) / 1.2) ** 2) * np.exp(-dist / 4)
p1 = expit(z0 + effect_logodds)
y0 = (rng.random(n) < p0).astype(int)
y1 = np.where(rng.random(n) < p1 / np.maximum(p0, 1e-12), y0, 0)  # a text never makes you later
ite = y1 - y0

show("share late without / with reminder", f"{y0.mean():.3f} / {y1.mean():.3f}")
show("true ATE  E[Y1 - Y0]", f"{ite.mean():+.3f}")
show("riders the text 'saves' (Y0=1, Y1=0)", f"{(ite == -1).sum():,} of {n:,}")
show("first five riders (Y0, Y1)", list(zip(y0[:5].tolist(), y1[:5].tolist())))

# ---------------------------------------------------------------------------
# 2. The fundamental problem: we only see one
# ---------------------------------------------------------------------------
print("\n2. What we actually observe")

# Last month operations texted riders they judged risky: the more past late
# rides, the more likely you got a text.
t = (rng.random(n) < expit(-2.0 + 0.6 * past_late)).astype(int)
y = np.where(t == 1, y1, y0)  # the switching equation
show("riders texted", f"{t.mean():.1%}")
show("first five riders (T, observed Y)", list(zip(t[:5].tolist(), y[:5].tolist())))

# ---------------------------------------------------------------------------
# 3. The naive comparison and where it goes wrong
# ---------------------------------------------------------------------------
print("\n3. The naive comparison")

naive = y[t == 1].mean() - y[t == 0].mean()
att = ite[t == 1].mean()
selection = y0[t == 1].mean() - y0[t == 0].mean()
show("late | texted  minus  late | not texted", f"{naive:+.3f}")
show("  = ATT  E[Y1 - Y0 | texted]", f"{att:+.3f}")
show("  + selection bias  E[Y0|T=1] - E[Y0|T=0]", f"{selection:+.3f}")
show("check: ATT + bias", f"{att + selection:+.3f}")

# If instead we flip a coin for each rider, T can't depend on Y0 at all.
t_r = rng.integers(0, 2, size=n)
y_r = np.where(t_r == 1, y1, y0)
diff_r = y_r[t_r == 1].mean() - y_r[t_r == 0].mean()
show("randomised: late | texted minus not", f"{diff_r:+.3f}   (true ATE {ite.mean():+.3f})")

# ---------------------------------------------------------------------------
# 4. A good predictor
# ---------------------------------------------------------------------------
print("\n4. A predictor of lateness")

# Fit a logistic regression of Y0 on the features, using riders who weren't
# texted (Newton's method in NumPy, as in the first course).
X = np.column_stack([np.ones(n), past_late, dist])


def fit_logistic(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ b)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        b += np.linalg.solve(H, g)
    return b


def auc(score, label):
    r = np.argsort(np.argsort(score)) + 1
    pos = label == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


b = fit_logistic(X[t == 0], y[t == 0])
risk = expit(X @ b)
show("coefficients (intercept, past_late, dist)", np.round(b, 2).tolist())
show("AUC for 'late without a text' (all riders)", f"{auc(risk, y0):.3f}")

# ---------------------------------------------------------------------------
# 5. Two policies with the same budget
# ---------------------------------------------------------------------------
print("\n5. Spend 4,000 texts")

budget = 4000
true_uplift = p0 - p1  # expected reduction in lateness, known only in simulation


def late_after(policy_idx):
    send = np.zeros(n, bool)
    send[policy_idx] = True
    return np.where(send, y1, y0).sum()


top_risk = np.argsort(-risk)[:budget]
top_uplift = np.argsort(-true_uplift)[:budget]
random_pick = rng.choice(n, budget, replace=False)
base = y0.sum()
show("late returns, nobody texted", f"{base:,}")
for label, idx in [("random 4,000", random_pick), ("top 4,000 by risk", top_risk), ("top 4,000 by uplift", top_uplift)]:
    show(f"  {label}: late returns prevented", f"{base - late_after(idx):,}")
show("mean past_late: risk pick / uplift pick", f"{past_late[top_risk].mean():.1f} / {past_late[top_uplift].mean():.1f}")

# ---------------------------------------------------------------------------
# 6. Uplift by risk band: what the model can't see
# ---------------------------------------------------------------------------
print("\n6. Who does the text actually help?")

bands = np.quantile(risk, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
which = np.clip(np.searchsorted(bands, risk, side="right") - 1, 0, 4)
for k in range(5):
    m = which == k
    show(f"  risk quintile {k + 1}: late w/o text, true uplift",
         f"{y0[m].mean():.3f}, {(y0[m] - y1[m]).mean():+.3f}")

# ---------------------------------------------------------------------------
# Figure: risk vs uplift, coloured by past late rides
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
idx = rng.choice(n, 3000, replace=False)
sc = axes[0].scatter(risk[idx], true_uplift[idx], c=past_late[idx], s=4, cmap="viridis")
axes[0].set_xlabel("predicted risk of being late (no text)")
axes[0].set_ylabel("true uplift from a text")
axes[0].set_title("High risk is not high uplift")
fig.colorbar(sc, ax=axes[0], label="past late rides (of 10)")
cuts = np.arange(500, n + 1, 500)
for label, order in [("by risk", np.argsort(-risk)), ("by uplift", np.argsort(-true_uplift))]:
    prevented = [true_uplift[order[:c]].sum() for c in cuts]
    axes[1].plot(cuts, prevented, label=label)
axes[1].set_xlabel("texts sent")
axes[1].set_ylabel("expected late returns prevented")
axes[1].set_title("Same budget, different targeting")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "uplift.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/uplift.png")
