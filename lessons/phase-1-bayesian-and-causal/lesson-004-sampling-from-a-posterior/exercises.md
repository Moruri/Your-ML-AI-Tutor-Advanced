# Lesson 004 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
mkdir -p my_work_004
cp lessons/phase-1-bayesian-and-causal/lesson-004-sampling-from-a-posterior/lesson.py my_work_004/
cd my_work_004
python lesson.py
```

---

## 1. Why the constant doesn't matter

`log_post_beta` leaves out the normalising constant of `Beta(5, 22)`,
and also the `comb(10, 3)` from the binomial. Why is the sampler still
correct? What would happen if you added `+ 100` to every value it
returns?

<details>
<summary>Check yourself</summary>

The sampler only ever uses `log_post(proposal) - log_post(x)`. Any
constant, including `log comb(10, 3)`, the Beta normaliser, or your
`+ 100`, appears in both terms and cancels. The chain would be
identical, draw for draw, given the same random seed. That's precisely
why MCMC works on posteriors whose normalising constant (the "evidence")
nobody can compute.

</details>

## 2. Throw away the repeats?

A colleague says the repeated values in the chain look like a bug and
wants to keep only the steps where the walk actually moved. What goes
wrong?

<details>
<summary>Check yourself</summary>

The repeats are how the chain gives extra weight to likely places. A
spot with high density rejects most proposals to leave it, so it gets
recorded many times. Removing repeats gives every visited spot equal
weight, which over-weights the unlikely regions the walk passed through
quickly, so the results are biased towards the tails. With a
well-tuned sampler on a gentle target like Beta(5, 22) the damage can be
small enough to miss, which is exactly what makes it dangerous on
harder models. Keep every recorded step, including the repeats.

</details>

## 3. 98% acceptance

The tiny-step chain accepted 98% of its proposals. Explain to someone
new why that's a bad sign rather than a good one.

<details>
<summary>Check yourself</summary>

It accepts almost everything because each proposal is barely different
from where it already is. Nearby points always have nearly the same
density, so the ratio is close to 1. The chain is "succeeding" at
taking steps that go nowhere. Its draws are so correlated that 19,000 of
them are worth about 22 independent ones. High acceptance isn't the
goal; exploring the whole posterior is.

</details>

## 4. Hands-on: Station B breaks the sampler

Station B had 300 late out of 1,000. In your copy, write
`log_post_logit_b`, the same as section 5's `log_post_logit` but with
300 and 700 in place of `late` and `n - late`. Run one chain with step
0.6 from a start of -2.0, then print the acceptance rate and
`ess(chain[2000:, 0])`.

Predict before you run: will step 0.6 still work well? Then find a
better step.

<details>
<summary>Check yourself</summary>

```python
def log_post_logit_b(x):
    z = x[0]
    p = 1 / (1 + np.exp(-z))
    return stats.norm.logpdf(z, -2.0, 0.6) + 300 * np.log(p) + 700 * np.log(1 - p)

for step in [0.6, 0.1]:
    c, a = metropolis(log_post_logit_b, -2.0, step, 20_000, rng)
    print(step, round(a, 2), round(ess(c[2000:, 0])))
```

With 1,000 rides the posterior is much narrower (a rate of about 0.297,
95% interval about [0.27, 0.33]), so step 0.6 overshoots. Acceptance
drops to roughly 0.15 and the ESS roughly halves. A step of about 0.1
brings acceptance back to around 0.6 and roughly doubles the ESS. The
lesson: the right step size depends on how wide the posterior is, so
it changes with the data. That's why real samplers tune themselves
during warm-up, and why you should always look at the diagnostics.
Your exact numbers will differ a little from run to run.

</details>

## 5. Optional puzzle: how much later, in plain rates?

Odds ratios are hard to explain. Using the rain model's draws (`kept` in
section 6), find the posterior of the late rate on a rainy day and on a
dry day, and of their difference. What's the chance rain adds more than
20 percentage points?

<details>
<summary>Check yourself</summary>

Transform each draw (section 5's bonus):

```python
s = lambda z: 1 / (1 + np.exp(-z))
alpha, beta = kept[:, :, 0].ravel(), kept[:, :, 1].ravel()
wet, dry = s(alpha + beta), s(alpha)
diff = wet - dry
print(wet.mean(), dry.mean(), np.quantile(diff, [0.025, 0.975]), (diff > 0.2).mean())
```

Roughly 0.42 on rainy days against 0.15 on dry ones, a difference of
about 27 points with a 95% interval of roughly [0.10, 0.45], and about a
78% chance the difference is over 20 points. "On rainy days, late
returns jump from about one in seven to about two in five" is a
sentence an operations team can plan around. Turning samples into
quantities people care about is half the value of the Bayesian
approach.

</details>
