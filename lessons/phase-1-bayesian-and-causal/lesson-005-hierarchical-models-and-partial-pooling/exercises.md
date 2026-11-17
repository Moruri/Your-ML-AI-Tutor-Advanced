# Lesson 005 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_005
cp lessons/phase-1-bayesian-and-causal/lesson-005-hierarchical-models-and-partial-pooling/lesson.py my_work_005/
cd my_work_005
python lesson.py
```

---

## 1. The leaderboard trap

Your manager wants a monthly "worst five stations" list, ranked by raw
late rate. Using sections 2 and 4, explain what that list will look like
month after month, and what you'd rank by instead.

<details>
<summary>Check yourself</summary>

It will be dominated by tiny stations, because small samples produce
extreme rates by chance (K at 2/5 = 40%, L at 1/3 = 33%). Next month a
different set of tiny stations will be on it, so the list will churn and
nobody will trust it. Rank by the partially pooled rate instead, or by
the probability that a station's rate is above some target. A station
then needs consistent evidence to reach the top, not one unlucky
afternoon.

</details>

## 2. What tau would do

Without running anything: what would the pooled rates look like if the
posterior for `tau` were squeezed close to 0? And if it were very large,
say 5?

<details>
<summary>Check yourself</summary>

With `tau` near 0 the model believes every station has the same rate, so
every pooled estimate collapses to roughly the complete-pooling value
(about 0.12). With `tau` around 5 stations are allowed to be wildly
different, the shared prior is almost flat, and each station's estimate
is close to its raw rate (with very wide intervals for the small ones).
Partial pooling isn't a fixed compromise; `tau` sets the dial, and the
data sets `tau`.

</details>

## 3. Why not just add a fixed prior?

Lesson 003 used a fixed `Beta(2, 15)` prior for every station. How is
that different from today's model?

<details>
<summary>Check yourself</summary>

A fixed prior also shrinks every station, but by an amount someone chose
in advance, towards a centre someone chose in advance. The hierarchical
model **learns** both the centre (`mu`) and the strength of shrinkage
(`tau`) from all the stations together. If the stations turned out to be
very different, it would shrink less; if very alike, more. It's still
Bayesian; we've just moved the prior's own settings up a level and let
the data inform them.

</details>

## 4. Hands-on: do it with the sampler

Paste lesson 004's `metropolis`, `r_hat` and `ess` functions into your
copy. Write a log posterior for all fourteen unknowns at once, with
`x = [mu, log_tau, z_1, ..., z_12]` (sampling `log_tau` keeps `tau`
positive; remember to add `log_tau` itself to the log posterior, the
Jacobian of the change of variable). Run four chains of 40,000 steps
with step 0.12 and compare the posterior mean rate of stations A and K
with section 4.

<details>
<summary>Check yourself</summary>

```python
def log_post_h(x):
    mu, log_tau, z = x[0], x[1], x[2:]
    tau = np.exp(log_tau)
    lp = stats.norm.logpdf(mu, -2, 1) + stats.halfnorm.logpdf(tau, scale=1) + log_tau
    lp += stats.norm.logpdf(z, mu, tau).sum()
    lp += (late * z - rides * np.logaddexp(0, z)).sum()   # binomial-logit
    return lp

starts = [np.r_[-2, np.log(0.5), np.full(12, -2.0)] + rng.normal(0, 0.5, 14) for _ in range(4)]
cs = np.array([metropolis(log_post_h, s, 0.12, 40_000, rng)[0] for s in starts])
kept = cs[:, 10_000:]
print([round(r_hat(kept[:, :, i]), 3) for i in [0, 1, 2, 12]])
print(expit(kept[:, :, 2]).mean(), expit(kept[:, :, 12]).mean())
```

Station A should come out near 0.11 and K near 0.18, matching the grid.
But look closer at the diagnostics. Acceptance looks healthy (about
0.35), yet `ess(kept[0, :, 1])` for `log_tau` is only in the tens from
30,000 draws, and on shorter runs its R-hat creeps above 1.01. When `tau` is small all
the `z_j` must squeeze together, and a single step size can't serve
both the wide and the narrow parts of the posterior (statisticians call
this shape **Neal's funnel**). This is exactly why real hierarchical
models use smarter samplers (Hamiltonian Monte Carlo in Stan, PyMC and
NumPyro) and often a "non-centred" rewrite, `z_j = mu + tau * e_j`.
Your exact numbers will vary from run to run.

</details>

## 5. Optional puzzle: a new station opens

Tomorrow a thirteenth station opens. Before it records a single ride,
what's your best guess for its late rate, and how uncertain is it? Use
the `(mu, tau)` grid posterior `w`.

<details>
<summary>Check yourself</summary>

Draw from the **population**, not from any one station: for each grid
cell, a new station's log-odds is `Normal(mu, tau)`.

```python
idx = rng.choice(w.size, 20_000, p=w.ravel())
z_new = rng.normal(M.ravel()[idx], T.ravel()[idx])
r = expit(z_new)
print(r.mean(), np.quantile(r, [0.025, 0.975]))
```

You should get a mean around 0.15 with a wide 95% interval, roughly 0.05
to 0.34. That interval combines two things: uncertainty about the
typical station (`mu`) and genuine variation between stations (`tau`).
This is a **prediction for a new group**, which only a hierarchical
model can give you, and it's the right starting prior for that station
once its rides start arriving.

</details>
