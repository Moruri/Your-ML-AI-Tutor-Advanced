# Lesson 003 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Most of this you can do with a pencil first. Predict, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
mkdir -p my_work_003
cp lessons/phase-1-bayesian-and-causal/lesson-003-conjugate-priors-you-can-finish-on-paper/lesson.py my_work_003/
cd my_work_003
python lesson.py
```

---

## 1. The flat prior, on paper

Using only the update rule, what is Station A's posterior under the flat
prior `Beta(1, 1)`, and what is its mean? Does it match lesson 002?

<details>
<summary>Check yourself</summary>

`Beta(1 + 3, 1 + 7) = Beta(4, 8)`, with mean `4 / 12 = 0.333`, exactly
the flat-prior posterior mean from lesson 002. Its 95% interval is about
[0.109, 0.610] and P(rate > 0.25) is about 0.713, again matching the
grid. That's why exercise 2 of lesson 002 found `(3 + 1) / (10 + 2)`:
the flat prior is one pseudo-late ride and one pseudo-on-time ride.

</details>

## 2. Which number goes on the dashboard?

The ops dashboard has room for one number per station. For Station A,
would you show the mode (0.160), the mean (0.185) or the raw rate
(0.300)? Why?

<details>
<summary>Check yourself</summary>

There's no single right answer, but the raw 0.300 is the worst choice:
it ignores everything we know about stations, and with ten rides it
jumps around a lot. Between the other two, the mean is usually better
for a dashboard. It's the best guess if you're penalised by squared
error, and it's also the probability that the *next* ride is late
(lesson 002, section 6). Whichever you pick, put the interval next to
it. A number without its uncertainty invites someone to over-read it.

</details>

## 3. How long until it's real?

Station A keeps running at exactly 30% late. Starting from the
`Beta(2, 15)` prior, after how many rides in total will the posterior
mean reach 0.25? Work it out on paper.

<details>
<summary>Check yourself</summary>

Solve `(2 + 0.3n) / (17 + n) = 0.25`. That gives
`2 + 0.3n = 4.25 + 0.25n`, so `0.05n = 2.25` and `n = 45`. After 45
rides the posterior mean is `(2 + 13.5) / 62 = 0.25`. Notice that a
mean of 0.25 still only gives P(rate > 0.25) of about a half, so if
your rule is "act when we're 90% sure it's above 25%", you'll need a
lot more rides than that.

</details>

## 4. Hands-on: the perfect station

Station D has had 12 rides and none were late. In your copy, compute its
posterior under the `Beta(2, 15)` prior. Before running, predict: is
the posterior mean 0? What is P(rate > 0.10)?

<details>
<summary>Check yourself</summary>

```python
describe("Station D", a0 + 0, b0 + 12)
show("Station D: P(rate > 0.10)", f"{stats.beta.sf(0.10, a0, b0 + 12):.3f}")
```

The posterior is `Beta(2, 27)`, with mean about 0.069 and a 95% interval
of roughly [0.009, 0.183]. P(rate > 0.10) is still about 0.22. Zero out
of twelve doesn't mean the rate is zero; it means it's probably low.
A raw rate of 0% would be wrong in a way that matters: a bad
"guaranteed on time" promise to customers. Small samples of rare events
are exactly where a prior earns its keep.

</details>

## 5. Optional puzzle: breakdowns per week

Bikes break down at Station A at some weekly rate. Your prior, from the
network, is `Gamma(shape=3, rate=1)` (mean 3 per week). Over the last
four weeks you saw 2, 5, 4 and 6 breakdowns. Using the table in the
README, what's the posterior, its mean, and P(weekly rate > 5)?

<details>
<summary>Check yourself</summary>

Shape `3 + (2 + 5 + 4 + 6) = 20`, rate `1 + 4 = 5`, so `Gamma(20, 5)`
with mean `20 / 5 = 4.0` per week. In scipy, Gamma takes `scale = 1 /
rate`:

```python
g = stats.gamma(20, scale=1 / 5)
print(g.mean(), g.ppf([0.025, 0.975]), g.sf(5))
```

The 95% interval is about [2.44, 5.93] and P(rate > 5) is about 0.13.
The data's own average is 4.25; the prior pulled it slightly towards 3.
If you're keen, the chance of 7 or more breakdowns *next* week (the
posterior predictive, a negative binomial) is about 0.13 too:
`stats.nbinom.sf(6, 20, 5 / 6)`.

</details>
