# Lesson 002 - Probability as belief

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 1 | Friday 2026-11-13

> **Goal:** Represent a belief as a distribution and update it with
> Bayes' rule on a small, honest example.

Time: about 45 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 001 left Station A in an awkward spot: three late returns out of
ten rides, a bootstrap interval from 0.00 to 0.60, and a rule that
wants to close it. The bootstrap struggles here because it only knows
those ten rides. We know more than that. We run a whole network of
stations, and most of them are late about one time in ten.

The Bayesian way of thinking uses that knowledge openly:

1. Write down what you believe **before** the data, as a distribution
   over every possible value. That's the **prior**.
2. Ask how likely the data is under each possible value. That's the
   **likelihood**.
3. Multiply them and rescale so it sums to one. That's the
   **posterior**: what you believe **after** the data.

That's Bayes' rule, and it's the whole idea. Today we do it with
counts, then on a grid of 1,001 candidate rates with plain NumPy, so
every step is a multiplication you can print.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-002-probability-as-belief/lesson.py
```

It takes a couple of seconds and saves one figure to
`output/beliefs.png` next to the script.

## 1. Bayes' rule with counts

Start with something you've seen before. 1% of M-Pesa top-ups are
fraud. A flag catches 95% of fraud and wrongly flags 5% of honest
top-ups. A top-up is flagged: how likely is it to be fraud?

```
  of 10,000 top-ups: fraud, flagged fraud        -> 100, 95
  honest but flagged                             -> 495
  P(fraud | flagged), counting                   -> 0.161
  P(fraud | flagged), Bayes' rule                -> 0.161
  after a second independent flag                -> 0.785
```

Only about one flag in six is real fraud. Imagine 10,000 top-ups: 95
fraud flags against 495 false alarms, because honest top-ups vastly
outnumber fraud. That's Bayes' rule with counts:

```
P(fraud | flag) = P(flag | fraud) * P(fraud) / P(flag)
```

The prior (1%) matters enormously. Forgetting it is called the
**base-rate fallacy**, and it's one of the most common mistakes people
make with any detector, from medical tests to spam filters.

The last line shows the habit that makes Bayes so useful: after the
first flag, 0.161 becomes the new prior. A second, independent check
that also says "fraud" lifts it to 0.785. **Yesterday's posterior is
today's prior.**

## 2. A belief is a distribution

Fraud was a yes/no question. A late-return *rate* can be any number
from 0 to 1, so a belief about it is a whole distribution. We put 1,001
candidate rates on a grid (`theta`, from 0.000 to 1.000) and give each
one a weight. The weights sum to one.

Two priors for Station A:

- **Flat:** every rate equally plausible. "I know nothing."
- **Network:** shaped like `theta**1 * (1 - theta)**14`. Most of the
  weight sits near 0.1, with a tail that rarely passes 0.3. That's what
  our other stations tell us.

```
  flat prior: mean, 95% interval                 -> 0.500  [0.025, 0.975]
  flat prior: P(rate > 0.25)                     -> 0.749
  network prior: mean, 95% interval              -> 0.118  [0.016, 0.302]
  network prior: P(rate > 0.25)                  -> 0.063
  each prior sums to                             -> 1.000, 1.000
```

Read the intervals as plain statements of belief: under the network
prior, before seeing any rides, we're 95% sure the rate is between
0.016 and 0.302. That kind of sentence is called a **credible
interval**, and it's what most people *think* a confidence interval
says (exercise 2 of lesson 001).

"Flat" is not as innocent as it sounds. It says a 90% late rate is as
plausible as a 10% one, which no one running a bike network believes.
Every prior is a choice; the honest thing is to make it out loud.

## 3. Update with the evidence

The likelihood of "3 late out of 10" for each candidate rate is the
binomial formula you met in the first course:

```python
likelihood = comb(10, 3) * theta**3 * (1 - theta)**7
posterior  = normalise(prior * likelihood)
```

That second line is Bayes' rule for all 1,001 rates at once.

```
  likelihood peaks at rate                       -> 0.300
  flat posterior: mean, 95% interval             -> 0.333  [0.109, 0.610]
  flat posterior: P(rate > 0.25)                 -> 0.712
  network posterior: mean, 95% interval          -> 0.185  [0.066, 0.349]
  network posterior: P(rate > 0.25)              -> 0.183
```

Open `output/beliefs.png` to see all three curves together: the prior
leaning towards 0.1, the likelihood centred on 0.3, and the posterior
sitting between them.

The data alone points at 0.30. The network posterior lands between the
prior and the data, at 0.185, because ten rides is real evidence but
not overwhelming evidence. This pulling-towards-what's-typical is
called **shrinkage**, and lesson 005 builds a whole model around it.

Now the decision is easy to state honestly. "Is Station A worse than
25%?" Under the network prior there's an 18% chance it is. That's
worth watching, not worth closing a station over. Notice how different
the flat prior's answer is (71%). With ten rides, the prior is doing a
lot of work, which is exactly why it should be a prior you can defend.

## 4. One ride at a time

Feed the same ten rides in one at a time, updating after each:

```
  after ride  1 (0 late): mean                   -> 0.111
  after ride  2 (1 late): mean                   -> 0.158
  after ride  5 (2 late): mean                   -> 0.182
  after ride 10 (3 late): mean                   -> 0.185
  max gap vs all-at-once posterior               -> 4.3e-18
```

Same answer, to within rounding error. Each step multiplies by `theta`
(late) or `1 - theta` (on time), and multiplication doesn't care about
order. So a Bayesian model can learn as data arrives, without keeping
every old ride around: the current posterior *is* the summary.

## 5. With lots of data, the prior stops mattering

Station B had 300 late out of 1,000:

```
  flat posterior: mean, 95% interval             -> 0.300  [0.272, 0.329]
  network posterior: mean, 95% interval          -> 0.297  [0.269, 0.325]
```

The two priors, so different at Station A, now agree almost exactly,
and both match the bootstrap from lesson 001. A thousand rides
outweigh any reasonable prior. That's reassuring: the prior matters
most when data is scarce, which is exactly when you'd want to use what
you already know.

## 6. Predict what happens next

The posterior isn't the end. Usually we want to predict:

```
  P(next ride at A is late)                      -> 0.185
  P(3+ late in next 10), full posterior          -> 0.291
  P(3+ late in next 10), plug in the mean        -> 0.278
```

The chance the next ride is late is just the posterior mean. For "3 or
more late out of the next 10", we average the binomial over every
candidate rate, weighted by the posterior. That's the **posterior
predictive**. Plugging in a single best rate gives a slightly smaller
number, because it pretends we know the rate exactly. The full version
carries our uncertainty into the prediction, so it's a bit wider. With
ten rides that gap is small; with three it would be large.

## What to remember

- A belief about an unknown quantity is a **distribution**, not a
  number.
- **Posterior is proportional to prior times likelihood.** On a grid,
  that's one multiplication and a rescale.
- **Base rates matter**, especially for rare events.
- The prior matters when data is scarce and fades as data piles up.
  Choose one you can defend, and say what it is.
- Predict with the **posterior predictive** so your uncertainty comes
  along.

A grid works beautifully for one unknown. With ten unknowns you'd need
`1001**10` grid points, which is why lesson 003 finds the exact
answer on paper for this model, and lesson 004 learns to *sample* when
paper runs out.

## What to do now

1. Run `lesson.py` and look at `output/beliefs.png`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks how strong a prior should be.
3. Then rest. Lesson 003 picks up on Monday.
