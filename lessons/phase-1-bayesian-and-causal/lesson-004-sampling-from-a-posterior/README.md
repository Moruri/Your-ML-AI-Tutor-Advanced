# Lesson 004 - Sampling from a posterior

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 2 | Monday 2026-11-16

> **Goal:** Implement a Metropolis-Hastings sampler, plot the chain, and
> decide whether it has mixed.

Time: about 50 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 003 ended with a trick: draw samples from a posterior and count.
We could draw them because scipy knew the exact posterior, `Beta(5, 22)`.
Real models rarely hand you that. Swap in a prior that isn't a Beta, or
add a second unknown, and the formula is gone. A grid still works for
one or two unknowns, but `1001**10` points for ten unknowns is out of
the question.

**Markov chain Monte Carlo** (MCMC) gets around this. Instead of writing
the posterior down, we build a random walk that spends time in each
region in proportion to its posterior probability. Record where it
goes, and the record *is* a sample. The oldest and simplest version,
**Metropolis-Hastings**, fits in fifteen lines. Modern tools (PyMC,
Stan, NumPyro) use smarter walks, but every one of them can fail in the
ways you'll see today, and you'll diagnose them the same way.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-004-sampling-from-a-posterior/lesson.py
```

It takes about ten seconds (plain Python loops, on purpose, so you can
read every step) and saves `output/chains.png` next to the script.

## 1. The sampler

Stand somewhere. Then repeat:

1. **Propose** a step: your current spot plus a little Normal noise.
2. If the proposal has higher posterior density, **move** there.
3. If it's lower, move there **with probability** `new / old`.
   Otherwise stay put, and record your current spot *again*.

```python
proposal = x + step * rng.standard_normal(x.size)
if np.log(rng.random()) < log_post(proposal) - log_post(x):
    x = proposal
chain[i] = x
```

Two details make this practical:

- **Only the ratio matters**, so the posterior's normalising constant
  cancels. We only need prior times likelihood, which we can always
  write down. That's the whole reason MCMC exists.
- **We work in logs.** `theta**300 * (1 - theta)**700` underflows to
  zero in floating point; its log is a perfectly ordinary number.

Our first target is Station A again, with the `Beta(2, 15)` prior and 3
late out of 10:

```
  first five values                              -> [0.435, 0.435, 0.271, 0.208, 0.208]
  acceptance rate (step 0.1)                     -> 0.62
```

See the repeats? Those are rejected proposals. They belong in the
chain: staying put is how the walk spends extra time in the likely
places.

## 2. Check against an answer we know

Never trust a sampler you haven't tested on a known answer. Lesson 003
gave us one:

```
  sampler: mean, 95% interval                    -> 0.187  [0.067, 0.350]
  exact:   mean, 95% interval                    -> 0.185  [0.066, 0.349]
  sampler: P(rate > 0.25)                        -> 0.191
```

Close, not exact. That's Monte Carlo error: we have a finite sample, and
neighbouring draws are correlated, so 19,000 draws carry less
information than 19,000 independent ones. (The exact tail is 0.184.) We
also threw away the first 1,000 steps as **warm-up**, the stretch where
the walk is still travelling in from its arbitrary start at 0.5.

## 3. Step size

The step size is the one knob. Here are three settings, each run for
20,000 steps:

```
  step 0.005: acceptance, ESS of 19,000          -> 0.98, 22
  step 0.1  : acceptance, ESS of 19,000          -> 0.61, 3,241
  step 1.5  : acceptance, ESS of 19,000          -> 0.06, 684
```

**ESS**, the effective sample size, answers "how many independent draws
is this chain worth?" It shrinks as neighbouring draws become more
correlated. The lesson's `ess()` adds up the chain's autocorrelations
until they die away.

- **Tiny steps** get accepted almost every time, which sounds great but
  isn't. The walk shuffles, and 19,000 draws are worth about 22.
- **Huge steps** mostly land in silly places (below 0 or above 0.6) and
  get rejected, so the chain sits still for long stretches.
- **The middle** explores well. For one unknown, an acceptance rate
  somewhere around 0.4 to 0.7 is a fine target; with many unknowns,
  aim lower, around 0.25.

Open `output/chains.png`. The top-left trace (step 0.005) wanders
slowly like a lost tourist, the top-right (0.1) looks like a "fuzzy
caterpillar", and the bottom-left (1.5) is a staircase of flat runs.
Learn to recognise all three by eye.

## 4. Has it mixed?

A single chain can look calm while being stuck. The standard defence is
to run **several chains from very different starting points** and check
that they agree. **R-hat** puts a number on that. It compares the spread
between chains (each split in half) with the spread within them. If
they've all found the same distribution, R-hat is about 1.00.

```
  4 chains, step 0.005, 400 steps: R-hat         -> 9.388
    chain means                                  -> [0.058, 0.282, 0.477, 0.765]
  4 chains, step 0.1, 20,000 steps: R-hat        -> 1.000
    chain means                                  -> [0.187, 0.185, 0.185, 0.186]
```

The short, timid chains never left their starting neighbourhoods, and
R-hat screams. The well-tuned chains agree to the third decimal place.
A common rule: **don't report anything until R-hat is below 1.01 for
every unknown and the ESS is in the hundreds at least.**

Neither check proves the chain has mixed: four chains can all miss the
same faraway region. But passing them is the minimum, and failing
them means stop.

## 5. A prior with no closed form

Now the real use. The network team describes stations on the
**log-odds** scale, and says a station's log-odds of a late return is
about `Normal(-2.0, 0.6)`. That's a sensible prior (log-odds of -2 is a
rate of about 12%), but it isn't a Beta, so lesson 003's shortcut is
gone. We sample on the log-odds scale and convert each draw back to a
rate:

```
  R-hat                                          -> 1.000
  sampler: mean rate, 95% interval               -> 0.182  [0.071, 0.345]
  sampler: P(rate > 0.25)                        -> 0.164
  grid check: mean rate, P(rate > 0.25)          -> 0.182, 0.164
```

There's still only one unknown, so a grid can check us, and it agrees.
Notice the bonus: to get the posterior of the *rate*, we just
transformed each sample. Any function of the unknowns gets its own
posterior for free that way.

## 6. Two unknowns: does rain make returns late?

Last, a model with two unknowns and no formula. We simulate 80 rides at
Station A, 30 on rainy days, and model

```
log-odds of late = alpha + beta * rain
```

This is logistic regression from the first course, made Bayesian:
`alpha` is the dry-day log-odds, `beta` is the extra log-odds rain adds,
and both get Normal priors. The sampler doesn't care that there are two
unknowns. Its steps just become two-dimensional.

```
  late on rainy / dry days                       -> 14/30, 6/50
  R-hat for alpha, beta                          -> 1.001, 1.002
  rain effect beta: mean, 95% interval           -> 1.46  [0.55, 2.39]
  P(rain makes lateness more likely)             -> 0.999
  odds ratio exp(beta): median                   -> 4.30
  ESS for beta, chain 1                          -> 854
```

The chains agree, and we're nearly certain rain makes lateness more
likely, with odds roughly 1.7 to 11 times higher (`exp` of the
interval ends). We set the true `beta` to 0.9 when we simulated. The
estimate is higher because this particular month happened to have a
wet streak of late returns, but 0.9 sits comfortably inside the 95%
interval. That's what an honest interval does: it doesn't always hit
the target dead centre, but it doesn't claim more certainty than the
data allows.

The bottom-right panel of `output/chains.png` shows the joint draws for
`alpha` and `beta`. They slope downwards: if dry days are a bit worse
than we think, rain needs to add a bit less to explain the rainy days.
Correlated unknowns like this are what slow a simple random walk down,
and why the ESS here is lower than in section 3.

## What to remember

- MCMC draws samples from a posterior you **can't normalise**: only the
  ratio of prior times likelihood matters.
- Metropolis-Hastings: **propose, accept with probability
  `min(1, ratio)`, otherwise stay.** Keep the repeats.
- **Test your sampler on a known answer** before trusting it on an
  unknown one.
- Tune the step size. Watch the **acceptance rate**, the **trace plot**
  and the **ESS**.
- Run **several chains from spread-out starts** and check **R-hat** is
  about 1.00 before you report anything.
- Any function of the samples (a rate, a difference, an odds ratio) is
  a sample from its own posterior.

## What to do now

1. Run `lesson.py` and look at all four panels of `output/chains.png`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   breaks the sampler on purpose and asks you to fix it.
3. Then rest. Lesson 005 uses everything from this week to let
   stations share what they know.
