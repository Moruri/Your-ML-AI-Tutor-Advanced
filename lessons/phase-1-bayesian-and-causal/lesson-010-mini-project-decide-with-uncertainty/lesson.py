"""
Lesson 010 - Mini-project: decide with uncertainty

The phase 1 finale. Product wants to know whether to offer every rider a
KES 200 weekend pass (free weekend rides for 28 days). We ran a proper A/B
test. Today we take it from raw data to a recommendation, using everything in
lessons 001-009: write the plan first, check the coin, estimate the effect
frequentist-style and Bayesian-style, refuse a tempting confounded comparison,
partially pool the per-city results, and turn a posterior into a decision with
costs attached.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/decision.png`, next to
this script (ignored by git), and runs in a few seconds.
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
rng = np.random.default_rng(11)


def show(label, value):
    print(f"  {label:<46} -> {value}")


def expit(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# 0. The plan, written before anyone looks at the results
# ---------------------------------------------------------------------------
PLAN = {
    "primary metric": "28-day revenue per rider (KES), everyone assigned",
    "primary analysis": "difference in means, adjusted for pre-period rides",
    "riders per arm": 10_000,
    "one-off launch cost (KES)": 6_000_000,
    "active riders per month": 150_000,
    "horizon (months)": 12,
}
COST, RIDERS, MONTHS = (PLAN["one-off launch cost (KES)"],
                        PLAN["active riders per month"], PLAN["horizon (months)"])
BREAK_EVEN = COST / (RIDERS * MONTHS)  # KES per rider-month needed to pay back
N = PLAN["riders per arm"]
PRICE, FARE = 200, 60

print("\n0. The plan")
for k, v in PLAN.items():
    show(k, f"{v:,}" if isinstance(v, int) else v)
show("break-even gain per rider-month", f"KES {BREAK_EVEN:.2f}")

# ---------------------------------------------------------------------------
# The world (you wouldn't see this part in real life)
# ---------------------------------------------------------------------------
CITIES = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika"]
SHARE = np.array([0.45, 0.2, 0.12, 0.1, 0.08, 0.05])
CITY_PACE = np.array([1.1, 0.9, 1.0, 0.95, 0.85, 1.0])


def riders(n):
    """Each rider has a personal ride rate; we see last month's rides."""
    city = rng.choice(len(CITIES), size=n, p=SHARE)
    lam = rng.gamma(1.5, 5 / 1.5, n) * CITY_PACE[city]
    pre = rng.poisson(lam)
    return city, lam, pre


def outcomes(lam, offered):
    """Revenue, rides and whether a pass was bought, over the 28 days."""
    bought = offered & (rng.random(len(lam)) < expit(-2.6 + 0.12 * lam))
    rate = np.where(bought, 1.25 * lam, lam)  # pass holders ride more
    rides = rng.poisson(rate)
    weekend = rng.binomial(rides, 0.5)  # half of rides fall on weekends
    revenue = np.where(bought, PRICE + FARE * (rides - weekend), FARE * rides)
    return revenue.astype(float), rides, bought


def expected_gain(lam):
    """What the offer is worth, on average, for a rider with ride rate lam."""
    p_buy = expit(-2.6 + 0.12 * lam)
    return p_buy * (PRICE + FARE * 0.5 * 1.25 * lam - FARE * lam)


# The true effects, overall and per city, from a very large population.
CITY_BIG, LAM, _ = riders(2_000_000)
GAIN = expected_gain(LAM)
TRUE = GAIN.mean()
TRUE_CITY = np.array([GAIN[CITY_BIG == c].mean() for c in range(len(CITIES))])

city, lam, pre = riders(2 * N + 137)  # a few extra sign-ups trickled in
t = rng.random(len(lam)) < 0.5  # the coin
rev, rides, bought = outcomes(lam, t)
n1, n0 = t.sum(), (~t).sum()

# ---------------------------------------------------------------------------
# 1. Check the coin before reading any results
# ---------------------------------------------------------------------------
print("\n1. Check the coin")
srm_p = stats.chisquare([n1, n0]).pvalue
show("riders: offer / control", f"{n1:,} / {n0:,}")
show("sample ratio mismatch p-value", f"{srm_p:.2f}")
smd = (pre[t].mean() - pre[~t].mean()) / np.sqrt((pre[t].var() + pre[~t].var()) / 2)
show("pre-period rides: offer / control", f"{pre[t].mean():.2f} / {pre[~t].mean():.2f}")
show("SMD of pre-period rides", f"{smd:+.3f}")

# ---------------------------------------------------------------------------
# 2. Frequentist estimates
# ---------------------------------------------------------------------------
print("\n2. Frequentist: estimate, interval, p-value")


def diff(y):
    a, b = y[t], y[~t]
    d = a.mean() - b.mean()
    return d, np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


d_raw, se_raw = diff(rev)
show("revenue per rider: offer / control", f"KES {rev[t].mean():.1f} / {rev[~t].mean():.1f}")
show("plain difference, 95% CI", f"{d_raw:+.2f}  [{d_raw - 1.96 * se_raw:+.2f}, {d_raw + 1.96 * se_raw:+.2f}]")
show("plain p-value", f"{2 * stats.norm.sf(abs(d_raw / se_raw)):.3f}")

# CUPED-style adjustment (lesson 007): subtract what last month predicts.
theta = np.cov(rev, pre)[0, 1] / pre.var(ddof=1)
rev_adj = rev - theta * (pre - pre.mean())
d, se = diff(rev_adj)
p_adj = 2 * stats.norm.sf(abs(d / se))
show("correlation of revenue with pre-period rides", f"{np.corrcoef(rev, pre)[0, 1]:.2f}")
show("adjusted difference, 95% CI", f"{d:+.2f}  [{d - 1.96 * se:+.2f}, {d + 1.96 * se:+.2f}]")
show("adjusted p-value", f"{p_adj:.3f}")
show("true effect (hidden in real life)", f"{TRUE:+.2f}")
show("pass take-up in offer arm", f"{bought[t].mean():.1%}")
d_r, se_r = diff(rides.astype(float))
show("rides per rider, difference", f"{d_r:+.2f}  (SE {se_r:.2f})")

# ---------------------------------------------------------------------------
# 3. The tempting comparison: pass holders vs everyone else
# ---------------------------------------------------------------------------
print("\n3. A comparison to refuse")
show("revenue: pass buyers / offered non-buyers", f"KES {rev[bought].mean():.0f} / {rev[t & ~bought].mean():.0f}")
show("pre-period rides: buyers / non-buyers", f"{pre[bought].mean():.1f} / {pre[t & ~bought].mean():.1f}")
show("'effect of buying', buyers minus control", f"{rev[bought].mean() - rev[~t].mean():+.0f}")

# ---------------------------------------------------------------------------
# 4. Bayesian: posterior for the effect, under two priors
# ---------------------------------------------------------------------------
print("\n4. Bayesian: posterior for the revenue effect")
# With 10,000 riders per arm the likelihood for the mean difference is very
# close to Normal(d, se). Combine it with a Normal prior (lesson 003 algebra).
# Bayesian bootstrap as a check that needs no normality (lesson 001's idea):
w1 = rng.dirichlet(np.ones(n1), size=1000)
w0 = rng.dirichlet(np.ones(n0), size=1000)
bb = w1 @ rev_adj[t] - w0 @ rev_adj[~t]
show("Bayesian bootstrap: mean, sd", f"{bb.mean():+.2f}, {bb.std():.2f}")


def posterior(prior_sd):
    if prior_sd is None:
        return d, se
    prec = 1 / prior_sd**2 + 1 / se**2
    return (d / se**2) / prec, np.sqrt(1 / prec)


draws = {}
for name, psd in [("flat", None), ("sceptical", 8.0)]:
    m, s = posterior(psd)
    draws[name] = rng.normal(m, s, 200_000)
    g = draws[name]
    show(f"{name}: mean, 95% interval", f"{m:+.2f}  [{np.quantile(g, 0.025):+.2f}, {np.quantile(g, 0.975):+.2f}]")
    show(f"{name}: P(effect > 0)", f"{(g > 0).mean():.3f}")
    show(f"{name}: P(effect > break-even)", f"{(g > BREAK_EVEN).mean():.3f}")

# Take-up, with a prior from past offers (about 10%, worth 40 riders).
a_post, b_post = 4 + bought[t].sum(), 36 + (t & ~bought).sum()
show("take-up posterior Beta(a, b)", f"Beta({a_post:,}, {b_post:,})")
show("take-up 95% interval", f"[{stats.beta.ppf(0.025, a_post, b_post):.3f}, {stats.beta.ppf(0.975, a_post, b_post):.3f}]")

# ---------------------------------------------------------------------------
# 5. City by city, with partial pooling
# ---------------------------------------------------------------------------
print("\n5. City by city")
est = np.array([rev_adj[(city == c) & t].mean() - rev_adj[(city == c) & ~t].mean() for c in range(6)])
ses = np.array([np.sqrt(rev_adj[(city == c) & t].var(ddof=1) / ((city == c) & t).sum()
                        + rev_adj[(city == c) & ~t].var(ddof=1) / ((city == c) & ~t).sum())
                for c in range(6)])
# Normal-normal hierarchical model (lesson 005): city effects ~ N(mu, tau^2).
# Like lesson 005, work on a grid of tau and AVERAGE over it, weighting each
# tau (flat prior on 0-20) by how well it explains the six estimates, with mu
# integrated out. Picking the single best tau instead can collapse towards 0
# ("all cities identical") just by chance.
taus = np.linspace(0.05, 20, 400)
ll, cond = [], []
for tau in taus:
    v = ses**2 + tau**2
    mu = (est / v).sum() / (1 / v).sum()  # best overall effect for this tau
    ll.append(stats.norm.logpdf(est, mu, np.sqrt(v)).sum() + 0.5 * np.log(1 / (1 / v).sum()))
    cond.append((est / ses**2 + mu / tau**2) / (1 / ses**2 + 1 / tau**2))
wt = np.exp(np.array(ll) - max(ll))
wt /= wt.sum()
shrunk = wt @ np.array(cond)
mu_hat = wt @ np.array([((est / (ses**2 + tau**2)).sum() / (1 / (ses**2 + tau**2)).sum()) for tau in taus])
tau_hat = wt @ taus
show("overall effect across cities, mu", f"{mu_hat:+.2f}")
show("spread between cities, tau (posterior mean)", f"{tau_hat:.2f}")
for c in range(6):
    show(f"  {CITIES[c]:<8} raw (SE) -> pooled [true]",
         f"{est[c]:+6.2f} ({ses[c]:4.1f}) -> {shrunk[c]:+6.2f}  [{TRUE_CITY[c]:+.2f}]")
rmse = lambda e: np.sqrt(((e - TRUE_CITY) ** 2).mean())
show("error vs truth (RMSE): raw / pooled", f"{rmse(est):.2f} / {rmse(shrunk):.2f}")

# ---------------------------------------------------------------------------
# 6. From posterior to decision
# ---------------------------------------------------------------------------
print("\n6. The decision")
g = draws["sceptical"]
value = (g * RIDERS * MONTHS - COST) / 1e6  # KES millions over the horizon
loss_ship = np.maximum(-value, 0).mean()
loss_hold = np.maximum(value, 0).mean()
show("expected value of shipping (KES m)", f"{value.mean():+.1f}")
show("P(shipping loses money)", f"{(value < 0).mean():.3f}")
show("expected loss if we ship (KES m)", f"{loss_ship:.2f}")
show("expected loss if we hold (KES m)", f"{loss_hold:.2f}")
show("decision", "ship" if loss_ship < loss_hold else "hold")
# Should we keep testing instead? Perfect information could at best save us
# the expected loss of our chosen action (the expected value of perfect
# information). Waiting a month costs a month of the expected gain.
show("most that more data could be worth (KES m)", f"{min(loss_ship, loss_hold):.2f}")
show("expected gain given up per month of waiting", f"{g.mean() * RIDERS / 1e6:.2f}")
# A second opinion from the frequentist side: the one-sided test of
# "effect <= break-even" at 5%.
z_be = (d - BREAK_EVEN) / se
show("one-sided p, effect <= break-even", f"{stats.norm.sf(z_be):.3f}")

# ---------------------------------------------------------------------------
# Figure: posterior with the break-even line, and the city forest plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, gg in draws.items():
    axes[0].hist(gg, bins=120, density=True, histtype="step", label=f"{name} prior")
axes[0].axvline(0, color="grey", lw=0.8)
axes[0].axvline(BREAK_EVEN, color="crimson", ls="--", label=f"break-even ({BREAK_EVEN:.2f})")
axes[0].set_xlabel("effect on 28-day revenue per rider (KES)")
axes[0].set_title("Posterior for the revenue effect")
axes[0].legend(fontsize=8)
ypos = np.arange(6)
axes[1].errorbar(est, ypos + 0.15, xerr=1.96 * ses, fmt="o", label="raw city estimate, 95% CI")
axes[1].plot(shrunk, ypos - 0.15, "s", color="C3", label="partially pooled")
axes[1].axvline(mu_hat, color="k", ls=":", label="overall")
axes[1].axvline(0, color="grey", lw=0.8)
axes[1].set_yticks(ypos, CITIES)
axes[1].set_xlabel("effect (KES per rider)")
axes[1].set_title("Cities: noisy alone, steadier together")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "decision.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/decision.png")
