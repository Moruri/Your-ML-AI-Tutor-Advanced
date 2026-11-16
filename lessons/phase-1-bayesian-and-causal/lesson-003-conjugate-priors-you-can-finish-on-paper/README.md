# Lesson 003 - Conjugate priors you can finish on paper

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 2 | Monday 2026-11-16

> **Goal:** Update a Beta-Binomial model by hand and in code, and read
> the posterior as a statement of uncertainty.

Time: about 40 minutes to read and run, plus the exercises. Nothing new
to install: `scipy` is already in `requirements.txt`.

---

On Friday we updated our belief about Station A on a grid of 1,001
candidate rates. That worked, and every step was visible, but it was
brute force. For this particular model there's a beautiful shortcut.

If your prior is a **Beta** distribution and your data is a count of
successes out of trials (a **Binomial**), then the posterior is
*another* Beta distribution. You get it by adding two numbers. Pairs
that fit together like this are called **conjugate**, and they're worth
knowing for three reasons: they're instant, they're exact, and they
give you a check for every fancier method you'll use later.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-003-conjugate-priors-you-can-finish-on-paper/lesson.py
```

It takes a couple of seconds and saves one figure to
`output/conjugate.png` next to the script.

## 1. The Beta distribution

A `Beta(a, b)` distribution lives between 0 and 1, which makes it a
natural belief about a rate. Its shape is

```
density  is proportional to  theta**(a - 1) * (1 - theta)**(b - 1)
```

Look back at lesson 002. The network prior was
`theta**1 * (1 - theta)**14`, which is exactly `Beta(2, 15)`. The flat
prior was `Beta(1, 1)`. We were using Betas all along without naming
them.

```
  network prior Beta(2, 15): mean, 95% interval  -> 0.118  [0.016, 0.302]
  flat prior Beta(1, 1): mean, 95% interval      -> 0.500  [0.025, 0.975]
  network prior: mode (a-1)/(a+b-2)              -> 0.067
  network prior: P(rate > 0.25)                  -> 0.063
```

Same numbers as Friday's grid, but now from formulas: the mean is
`a / (a + b)`, and `scipy.stats.beta` gives the interval and tail
probabilities in one call each.

## 2. The update by hand

Here's the whole rule:

```
prior      Beta(a, b)
data       k late out of n
posterior  Beta(a + k, b + n - k)
```

Add the late rides to `a`, the on-time rides to `b`. That's it. Why it
works takes one line: the likelihood is `theta**k * (1 - theta)**(n-k)`,
and multiplying it by the prior just adds the exponents.

```
  Station A: 3 late, 7 on time                   -> Beta(2+3, 15+7) = Beta(5, 22)
  posterior: mean, 95% interval                  -> 0.185  [0.066, 0.349]
  grid posterior mean (lesson 002)               -> 0.185
  max gap, grid vs Beta(5, 22)                   -> 1.0e-17
```

Friday's 1,001-point grid and today's two additions agree to the
seventeenth decimal place. You could have done this on the back of a
matatu ticket.

## 3. Reading the posterior

A posterior is a full answer, so you can ask it many questions. Each
one is a single line of code:

```
  mode (single most plausible rate)              -> 0.160
  standard deviation                             -> 0.073
  50% interval                                   -> [0.131, 0.231]
  P(rate > 0.25)                                 -> 0.184
  P(rate < 0.10)                                 -> 0.112
  rate we're 90% sure it's below                 -> 0.284
```

Each of those turns into a sentence you could say to the operations
lead:

- "Our best single guess is about 16-19% late."
- "It's an even bet that the rate is between 13% and 23%."
- "There's roughly an 18% chance it's worse than our 25% threshold."
- "We're 90% sure it's below 28%."

(Friday's grid said 0.183 for the tail; the exact answer is 0.184. The
grid was very slightly coarse, which is a nice reminder that the
formula is the ground truth here.)

Notice the mode (0.160) sits below the mean (0.185). The posterior has
a longer tail to the right, because rates near zero are squeezed
against a wall and high rates aren't. For skewed beliefs, say which
summary you're quoting.

## 4. The prior is pseudo-data

The neatest way to think about `Beta(a, b)` is as rides you've
already seen: `a` late ones and `b` on-time ones. `Beta(2, 15)` is
worth 17 rides, 2 of them late. Then the posterior mean is just a
weighted average of the prior mean and the data:

```
posterior mean = w * (k / n) + (1 - w) * (a / (a + b)),   w = n / (n + a + b)
```

```
  prior worth this many rides                    -> 17
  weight on the data, n / (n + a + b)            -> 0.370
  blend of 0.300 and 0.118                       -> 0.185
  after   10 rides at 30% late: mean             -> 0.185
  after  100 rides at 30% late: mean             -> 0.274
  after 1000 rides at 30% late: mean             -> 0.297
```

With ten rides, the data gets 37% of the say. With a thousand, it gets
almost all of it. This is the "prior fades as data piles up" idea from
lesson 002, now with an exact formula. It also gives you a quick sanity
check on any prior: *how many rides is this worth, and do I really have
that much prior knowledge?*

## 5. Choosing a prior from the network

So far we've taken `Beta(2, 15)` on trust. Where should a prior come
from? Ideally from data you already have. Here we pretend we have last
quarter's late rates at 24 other stations (simulated, but realistic),
and pick the Beta with the same mean and spread. That's the **method of
moments**:

```
strength = mean * (1 - mean) / variance - 1
a = mean * strength
b = (1 - mean) * strength
```

```
  24 stations: mean, sd of late rate             -> 0.112, 0.063
  matching Beta(a, b)                            -> Beta(2.7, 21.4), worth 24 rides
  Station A with that prior: mean, 95% interval  -> 0.167  [0.064, 0.307]
```

The fitted prior is a little more confident than our hand-picked one,
so Station A shrinks a little more. `a` and `b` don't have to be whole
numbers.

One caveat to keep in mind. The spread of 24 *observed* rates includes
the noise from each station's own small sample, so this prior is a
little too wide. Fixing that properly, by learning the prior and every
station's rate together, is exactly what lesson 005 does.

## 6. Is Station A worse than Station C?

Station C had 9 late out of 60. Same prior, separate updates. Now ask a
question with no neat formula: what's the chance A's true rate is higher
than C's?

```
  Station C, Beta(11, 66): mean, 95% interval    -> 0.143  [0.075, 0.229]
  P(A worse than C), from 100k draws             -> 0.679
  difference A - C: mean                         -> 0.043
  difference A - C: 95% interval                 -> [-0.107, 0.220]
```

Rather than wrestle with integrals, we drew 100,000 plausible rates from
each posterior and simply counted how often A's draw beat C's. About
two times in three. The interval for the difference comfortably
includes zero, so "A is worse" is a hunch, not a finding.

Hold on to that trick. Drawing samples and counting is how you answer
almost any question about a posterior. Here scipy could draw the samples
for us because we knew the posterior exactly. The next model won't be
so tidy, and lesson 004 builds a sampler that works even when there's no
formula at all.

`output/conjugate.png` shows the prior, both stations' posteriors, and
Friday's grid points sitting right on top of the exact curve.

## Other conjugate pairs you'll meet

You don't need to memorise these, but it helps to know they exist:

| Data | Prior | Update |
|------|-------|--------|
| Successes out of trials (Binomial) | Beta(a, b) | Beta(a + k, b + n - k) |
| Counts per period (Poisson) | Gamma(shape, rate) | Gamma(shape + total count, rate + periods) |
| Category counts (Multinomial) | Dirichlet(a1, ..., aK) | add each category's count to its `a` |
| Measurements with known noise (Normal) | Normal | precision-weighted average of prior and data |

The optional puzzle in the exercises tries the Gamma-Poisson one.

## What to remember

- **Beta prior + Binomial data = Beta posterior.** Add late counts to
  `a`, on-time counts to `b`.
- A posterior answers many questions: means, modes, intervals, tail
  probabilities. Say which one you're quoting.
- `Beta(a, b)` is worth `a + b` observations. The posterior mean is a
  weighted blend of prior and data.
- A prior should come from real knowledge, and you can fit one to past
  data.
- When there's no formula for your question, **draw samples and
  count**.

## What to do now

1. Run `lesson.py` and look at `output/conjugate.png`.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   meets a station with zero late returns.
3. Then go straight on to lesson 004, where we build the sampler.
