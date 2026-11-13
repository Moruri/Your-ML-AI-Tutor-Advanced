"""
Lesson 001 - Why this track exists, and how we'll learn

The first lesson of the advanced track. No new technique today. Instead,
three small simulations that show the gaps this track fills, each one a
question the first course taught you to ask but not yet to answer:

  1. How sure are we?        Same average, very different evidence.
  2. What caused what?       A great predictor that is a bad policy.
  3. Is it worth it?         A "better" model that isn't, once you count.

Then a quick check that your environment is ready for Phase 1.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. Everything is simulated, nothing is downloaded, and
it takes a second or two.
"""

import sys

import numpy as np

rng = np.random.default_rng(1)


def show(label, value):
    print(f"  {label:<46} -> {value}")


# ---------------------------------------------------------------------------
# 1. How sure are we? Same rate, different evidence
# ---------------------------------------------------------------------------
print("\n1. How sure are we?")

# Two new stations. Both have a 30% late-return rate so far.
stations = {"Station A (10 rides)": (3, 10), "Station B (1,000 rides)": (300, 1000)}
for name, (late, n) in stations.items():
    rides = np.r_[np.ones(late), np.zeros(n - late)]
    boots = rng.choice(rides, size=(5000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    show(f"{name}: late rate, 95% bootstrap", f"{late / n:.2f}  [{lo:.2f}, {hi:.2f}]")

# A naive rule: "close any station with a late rate above 25%".
show("naive rule closes both?", "yes, both are 0.30")
show("but could A really be below 25%?", "easily: its interval reaches 0.00")

# ---------------------------------------------------------------------------
# 2. What caused what? A good predictor, a bad policy
# ---------------------------------------------------------------------------
print("\n2. What caused what?")

# Simulated riders. Engagement (hidden) drives BOTH whether a rider turns on
# ride reminders AND whether they keep riding next month. Reminders themselves
# do nothing at all: their true effect is zero by construction.
n = 20_000
engaged = rng.random(n) < 0.4
reminders = rng.random(n) < np.where(engaged, 0.7, 0.2)
stays = rng.random(n) < np.where(engaged, 0.85, 0.45)

with_r, without_r = stays[reminders].mean(), stays[~reminders].mean()
show("stay rate, reminders on", f"{with_r:.3f}")
show("stay rate, reminders off", f"{without_r:.3f}")
show("observed gap (looks like an effect)", f"{with_r - without_r:+.3f}")

# Now run the experiment: switch reminders on for a random half of riders.
assigned = rng.random(n) < 0.5
stays_rct = rng.random(n) < np.where(engaged, 0.85, 0.45)  # reminders change nothing
gap_rct = stays_rct[assigned].mean() - stays_rct[~assigned].mean()
show("randomised gap (the real effect)", f"{gap_rct:+.3f}")

# Within each engagement group, the observed gap mostly vanishes too.
for flag, label in [(True, "engaged"), (False, "not engaged")]:
    m = engaged == flag
    g = stays[m & reminders].mean() - stays[m & ~reminders].mean()
    show(f"gap inside '{label}' riders", f"{g:+.3f}")

# ---------------------------------------------------------------------------
# 3. Is it worth it? Count what the model costs and saves
# ---------------------------------------------------------------------------
print("\n3. Is it worth it?")

# Two fraud flaggers for M-Pesa top-ups, scored on 10,000 simulated top-ups
# with 1% fraud. Each fraud caught saves KES 2,000; each false alarm costs
# KES 150 of a person's time and an annoyed customer.
n, fraud_rate = 10_000, 0.01
fraud = rng.random(n) < fraud_rate
models = {
    "simple rules": (0.70, 0.020),  # (catch rate, false-alarm rate)
    "big model": (0.80, 0.060),
}
for name, (tpr, fpr) in models.items():
    flagged = np.where(fraud, rng.random(n) < tpr, rng.random(n) < fpr)
    acc = (flagged == fraud).mean()
    caught = (flagged & fraud).sum()
    false_alarms = (flagged & ~fraud).sum()
    value = caught * 2000 - false_alarms * 150
    show(f"{name}: caught / false alarms", f"{caught} / {false_alarms}")
    show(f"{name}: accuracy, net value (KES)", f"{acc:.3f}, {value:,}")

# ---------------------------------------------------------------------------
# 4. Is your environment ready?
# ---------------------------------------------------------------------------
print("\n4. Environment check")
show("python", sys.version.split()[0])
for mod in ["numpy", "pandas", "scipy", "matplotlib", "sklearn"]:
    try:
        version = __import__(mod).__version__
    except ImportError:
        version = "MISSING - run: pip install -r requirements.txt"
    show(mod, version)
print("\nDone. Phase 1 needs only the packages above.")
