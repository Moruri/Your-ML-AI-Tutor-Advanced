"""
Lesson 008 - Confounding, graphs and what to adjust for

Do e-bikes make trips shorter? Riders choose them, and they choose them for a
reason: long trips and hilly stations. Those same things make trips longer.
To get the effect of the bike itself we have to adjust for some variables,
and, just as importantly, NOT adjust for others.

Today we write our assumptions down as a small causal graph, let a few lines
of code read off which paths are open, and check every answer against a
simulation where we know the truth.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/adjustment.png`, next
to this script (ignored by git), and runs in a second or two.
"""

from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(8)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def expit(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 1. The graph: our assumptions, written down
# ---------------------------------------------------------------------------
print("\n1. The causal graph")

# Each variable lists its direct causes (parents). This dict IS the model.
PARENTS = {
    "dist": [],                      # trip distance (km)
    "hill": [],                      # station on a hill
    "rain": [],                      # raining at pickup
    "promo": [],                     # e-bike discount shown in the app
    "ebike": ["dist", "hill", "promo"],
    "speed": ["ebike", "hill"],      # average speed (km/h)
    "duration": ["dist", "hill", "speed", "rain"],
    "flagged": ["ebike", "duration"],  # ride flagged for review
}
T, Y = "ebike", "duration"
for child, parents in PARENTS.items():
    if parents:
        print(f"    {child:<9} <- " + ", ".join(parents))


def children(node):
    return [c for c, ps in PARENTS.items() if node in ps]


def descendants(node):
    out, stack = set(), [node]
    while stack:
        for c in children(stack.pop()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def paths(a, b, path=None):
    """Every route from a to b, ignoring arrow direction, never revisiting."""
    path = path or [a]
    if a == b:
        yield path
        return
    for nxt in chain(PARENTS[a], children(a)):
        if nxt not in path:
            yield from paths(nxt, b, path + [nxt])


def arrow(u, v):
    return "->" if u in PARENTS[v] else "<-"


def is_open(path, Z):
    """d-separation for one path: is it open given adjustment set Z?"""
    for a, m, b in zip(path, path[1:], path[2:]):
        collider = a in PARENTS[m] and b in PARENTS[m]
        if collider and not ({m} | descendants(m)) & Z:
            return False  # an unconditioned collider blocks the path
        if not collider and m in Z:
            return False  # conditioning on a chain or fork blocks it
    return True


def describe(path):
    return " ".join(f"{u} {arrow(u, v)}" for u, v in zip(path, path[1:])) + f" {path[-1]}"


ALL = list(paths(T, Y))
show("paths between ebike and duration", len(ALL))
for p in ALL:
    kind = "causal" if all(arrow(u, v) == "->" for u, v in zip(p, p[1:])) else "non-causal"
    print(f"    {describe(p):<44} {kind:<10} open with no adjustment: {is_open(p, set())}")

# ---------------------------------------------------------------------------
# 2. Simulate the world the graph describes
# ---------------------------------------------------------------------------
print("\n2. Simulated rides")

n = 20_000
dist = rng.gamma(2.0, 1.5, size=n)
hill = (rng.random(n) < 0.3).astype(float)
rain = (rng.random(n) < 0.25).astype(float)
promo = (rng.random(n) < 0.5).astype(float)
ebike = (rng.random(n) < expit(-2.0 + 0.5 * dist + 1.5 * hill + 3.0 * promo)).astype(float)
speed = 12 + 8 * ebike - 2 * hill + rng.normal(0, 1.5, n)
duration = 6 + 4 * dist + 3 * hill - 0.5 * (speed - 12) + 6 * rain + rng.normal(0, 4, n)
flagged = (rng.random(n) < expit(-7 + 3 * ebike + 0.3 * duration)).astype(float)
DATA = dict(dist=dist, hill=hill, rain=rain, promo=promo, ebike=ebike,
            speed=speed, duration=duration, flagged=flagged)

# The bike adds 8 km/h and each km/h saves half a minute: -4 minutes, all of
# it through speed. That's the number every method below is trying to find.
TRUE_EFFECT = -0.5 * 8
show("share of rides on an e-bike", f"{ebike.mean():.1%}")
show("true effect of an e-bike on duration", f"{TRUE_EFFECT:+.1f} min")
show("mean dist: e-bike / pedal", f"{dist[ebike == 1].mean():.2f} / {dist[ebike == 0].mean():.2f} km")
show("share on a hill: e-bike / pedal", f"{hill[ebike == 1].mean():.2f} / {hill[ebike == 0].mean():.2f}")

# ---------------------------------------------------------------------------
# 3. Naive comparison, then adjustment, one variable at a time
# ---------------------------------------------------------------------------
print("\n3. Adjusting for confounders")


def effect(Z, rows=slice(None)):
    """OLS of duration on ebike plus the variables in Z; return coef and SE."""
    X = np.column_stack([np.ones(n), ebike] + [DATA[z] for z in Z])[rows]
    y = duration[rows]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    cov = r @ r / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)
    return b[1], np.sqrt(cov[1, 1])


def verdict(Z):
    """Backdoor criterion: no descendants of T in Z, and every non-causal path blocked."""
    if set(Z) & descendants(T):
        bad = sorted(set(Z) & descendants(T))
        return "no: " + ", ".join(bad) + " is downstream of ebike"
    leaks = [p for p in ALL if arrow(p[0], p[1]) == "<-" and is_open(p, set(Z))]
    return "yes" if not leaks else "no: open " + describe(leaks[0])


SETS = [
    [],
    ["dist"],
    ["dist", "hill"],
    ["dist", "hill", "rain"],
    ["dist", "hill", "promo"],
    ["dist", "hill", "speed"],
    ["dist", "hill", "flagged"],
]
results = []
for Z in SETS:
    est, se = effect(Z)
    results.append((Z, est, se))
    name = "{" + ", ".join(Z) + "}"
    show(f"adjust for {name}", f"{est:+.2f} (SE {se:.3f})   valid? {verdict(Z)}")

# ---------------------------------------------------------------------------
# 4. The mediator: adjusting for how the effect happens
# ---------------------------------------------------------------------------
print("\n4. Mediators")

for Z in (["dist", "hill"], ["dist", "hill", "speed"]):
    p_open = [describe(p) for p in ALL if arrow(p[0], p[1]) == "->" and is_open(p, set(Z))]
    show("causal paths open, adjusting {" + ", ".join(Z) + "}", p_open or "none")

# ---------------------------------------------------------------------------
# 5. The collider: selecting on a common effect
# ---------------------------------------------------------------------------
print("\n5. Colliders")

show("share of rides flagged", f"{flagged.mean():.1%}")
est_f, se_f = effect(["dist", "hill"], rows=flagged == 1)
show("{dist, hill}, flagged rides only", f"{est_f:+.2f} (SE {se_f:.3f})")
show("ebike -> flagged <- duration open, Z={}", is_open(["ebike", "flagged", "duration"], set()))
show("ebike -> flagged <- duration open, Z={flagged}", is_open(["ebike", "flagged", "duration"], {"flagged"}))

# ---------------------------------------------------------------------------
# 6. Precision vs noise: rain and promo
# ---------------------------------------------------------------------------
print("\n6. Variables that change the error bar, not the answer")

se_base = results[2][2]
for Z, est, se in results[3:5]:
    show("SE with " + Z[-1] + " vs without", f"{se:.3f} vs {se_base:.3f} ({se / se_base - 1:+.0%})")

# ---------------------------------------------------------------------------
# Figure: the graph, and every estimate against the truth
# ---------------------------------------------------------------------------
POS = {"promo": (0, 2), "dist": (1, 3), "hill": (3, 3), "ebike": (1, 1.5), "speed": (2.3, 1.5),
       "duration": (4.0, 1.5), "rain": (5.0, 2.7), "flagged": (2.3, 0)}
ROLE = {"dist": "tab:red", "hill": "tab:red", "speed": "tab:orange", "flagged": "tab:purple",
        "rain": "tab:green", "promo": "tab:gray", "ebike": "tab:blue", "duration": "tab:blue"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
for child, parents in PARENTS.items():
    for par in parents:
        ax.annotate("", xy=POS[child], xytext=POS[par],
                    arrowprops=dict(arrowstyle="->", lw=1.2, shrinkA=14,
                                    shrinkB=30 if POS[child][1] == POS[par][1] else 16))
for node, (x, yy) in POS.items():
    ax.text(x, yy, node, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec=ROLE[node], lw=2))
ax.set_xlim(-0.6, 5.6)
ax.set_ylim(-0.5, 3.5)
ax.axis("off")
ax.set_title("red: confounder  orange: mediator  purple: collider\n"
             "green: outcome-only  grey: treatment-only", fontsize=9)
ax = axes[1]
labels = ["{" + ", ".join(Z) + "}" for Z, _, _ in results] + ["{dist, hill}, flagged only"]
ests = [e for _, e, _ in results] + [est_f]
ses = [s for _, _, s in results] + [se_f]
ypos = np.arange(len(labels))[::-1]
ax.errorbar(ests, ypos, xerr=1.96 * np.array(ses), fmt="o", capsize=3)
ax.axvline(TRUE_EFFECT, color="k", ls="--", label="true effect")
ax.set_yticks(ypos, labels)
ax.set_xlabel("estimated effect of an e-bike on duration (min)")
ax.set_title("Same data, different adjustment sets")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "adjustment.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/adjustment.png")
