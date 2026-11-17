# Lesson 005 - Hierarchical models and partial pooling

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 3 | Tuesday 2026-11-17

> **Goal:** Share statistical strength across groups so rare ones shrink
> and common ones stay sharp.

Time: about 50 minutes to read and run, plus the exercises. Nothing new
to install.

---

So far we've studied one station at a time. Real networks have dozens,
and they are very uneven: a busy hub logs hundreds of rides a month, a
new kiosk at the edge of town logs three. If one of those three rides
comes back late, is that kiosk a 33% problem?

Two easy answers are both wrong. Judge every station only on its own
rides and the tiny ones swing wildly. Lump every station together and
you erase the real differences between them. A **hierarchical model**
does something in between: every station gets its own rate, but those
rates come from a shared distribution, and the data teaches us what that
distribution looks like. Small stations **borrow strength** from the
group. Big stations mostly speak for themselves. That's **partial
pooling**, and it's one of the most useful ideas in applied statistics.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-005-hierarchical-models-and-partial-pooling/lesson.py
```

It takes a few seconds and saves `output/shrinkage.png` next to the
script.

## 1. Twelve stations

We simulate twelve stations whose true late rates come from a
log-odds `Normal(-1.8, 0.5)`, so most sit between about 7% and 21%. We
keep the true rates so we can mark everyone's homework later, which you
can never do in real life.

```
  rides per station                              -> [400, 250, 180, 120, 90, 60, 40, 25, 15, 8, 5, 3]
  late returns                                   -> [43, 21, 16, 21, 18, 12, 2, 3, 5, 1, 2, 1]
```

## 2. No pooling

The obvious estimate is each station's own rate:

```
  raw rates                                      -> [0.11, 0.08, 0.09, 0.18, 0.2, 0.2, 0.05, 0.12, 0.33, 0.12, 0.4, 0.33]
  smallest three stations                        -> 1/8, 2/5, 1/3
```

Station K, with 2 late out of 5, looks like the worst in the network at
40%. Station G looks the best at 5%. Notice where the extremes are: at
the small end. That isn't because small stations are special. It's
because small samples are noisy, and noise pushes estimates outwards.
Rank stations on raw rates and your "best" and "worst" lists will be
full of tiny stations every single month.

## 3. Complete pooling

The opposite extreme: pretend all stations are the same.

```
  one shared rate                                -> 0.121
```

Steady, but it says Station D (21 of 120) and Station B (21 of 250) are
identical, which the data clearly doesn't support.

## 4. Partial pooling

The hierarchical model has two levels:

```
mu  ~ Normal(-2, 1)          typical station's log-odds
tau ~ HalfNormal(1)          how much stations differ from each other
z_j ~ Normal(mu, tau)        station j's own log-odds
late_j ~ Binomial(rides_j, expit(z_j))
```

The key unknown is **`tau`**. If `tau` is near zero, stations are all
alike and the model behaves like complete pooling. If `tau` is huge,
stations have nothing to do with each other and it behaves like no
pooling. We don't choose: the data estimates `tau`, and so it decides
how much pooling to do.

That's fourteen unknowns, too many for one big grid. But once `mu` and
`tau` are fixed, each station is a one-unknown problem, so the lesson
integrates each `z_j` out on a fine grid and is left with an exact
posterior over `(mu, tau)` on a 2-D grid. No sampler to tune. (Exercise
4 does it with lesson 004's sampler instead.)

```
  posterior mean of mu (as a rate)               -> 0.137
  posterior of tau: mean, 95% interval           -> 0.49  [0.14, 1.00]
  partially pooled rates                         -> [0.11, 0.09, 0.1, 0.16, 0.18, 0.17, 0.1, 0.13, 0.2, 0.14, 0.18, 0.16]
    station A (43/400): raw -> pooled            -> 0.11 -> 0.11  [0.08, 0.14]
    station J (1/8): raw -> pooled               -> 0.12 -> 0.14  [0.05, 0.28]
    station K (2/5): raw -> pooled               -> 0.40 -> 0.18  [0.08, 0.40]
    station L (1/3): raw -> pooled               -> 0.33 -> 0.16  [0.06, 0.36]
```

Station A barely moves: 400 rides is plenty of evidence. Station K
falls from 40% to 18%, and its interval is wide, honestly saying "we
don't know much yet". The model found `tau` around 0.5, which matches
the truth we simulated. It learned how different stations really are
from the stations themselves.

Open `output/shrinkage.png`. Each line joins a station's raw rate (red)
to its pooled rate (blue), with dot size showing rides. Small dots
travel a long way, big dots hardly move, and nobody lands exactly on the
dashed complete-pooling line.

## 5. Which estimate was closest?

Because we simulated, we can compare each method with the true rates:

```
  no pooling: error (all / small 6)              -> 0.083 / 0.143
  complete pooling: error (all / small 6)        -> 0.049 / 0.055
  partial pooling: error (all / small 6)         -> 0.035 / 0.049
```

(Mean absolute error, overall and for the six smallest stations.)
Partial pooling wins, and no pooling loses badly on the small stations.

One simulation could be luck, so the lesson repeats the whole story 200
times. To keep that fast it uses a cheaper relative of the full model,
**empirical Bayes**: fit a Beta prior to the group by the method of
moments (lesson 003), subtracting the part of the spread that's just
binomial noise, then do a conjugate update for each station.

```
  200 repeats: mean abs error (none/full/EB)     -> [0.059, 0.051, 0.035]
  EB beat no pooling in                          -> 96% of repeats
```

This is a real, repeatable effect, not a fluke. It's a cousin of a
famous result (Stein's paradox): when you estimate many related
quantities at once, pulling them towards each other beats estimating
each separately, on average.

Empirical Bayes is a fine tool, and you'll meet it in ranking systems
everywhere. The full model is better when there are few groups, because
it carries the uncertainty about `tau` into every station's interval
instead of plugging in one number.

## 6. Shrinkage depends on data

On the log-odds scale, each station's estimate is roughly a weighted
average of its own data and the group, with weight
`n_j / (n_j + kappa)`. Here `kappa` is about `1 / (tau² * m * (1 - m))`,
where `m` is the typical rate.

```
    station A (400 rides): weight on own data    -> 0.92
    station E (90 rides): weight on own data     -> 0.72
    station I (15 rides): weight on own data     -> 0.30
    station L (3 rides): weight on own data      -> 0.08
  rides needed to trust yourself half the time   -> 35
```

That last line is a nice number to tell a stakeholder: in this network,
a station needs about 35 rides before its own record counts for more
than the network's. It isn't a rule anyone picked. It comes from how
different the stations turned out to be.

## When to reach for this

Any time you have **groups of very different sizes and you care about
each group**: shop conversion rates, per-doctor outcomes, per-product
ratings, per-school exam results, per-customer churn, the effect of a
treatment in each region. The groups also need to be genuinely
comparable. Pooling a bike kiosk with a car-rental desk would drag both
somewhere meaningless.

## What to remember

- **No pooling** overreacts to small groups; **complete pooling**
  ignores real differences.
- A **hierarchical model** gives each group its own parameter, drawn
  from a shared distribution whose spread (`tau`) is learned from the
  data.
- Small groups **shrink** towards the group mean; large groups barely
  move. How much depends on `n_j` and `tau`.
- Shrunk estimates are **more accurate on average**, and their wide
  intervals are honest about small groups.
- **Empirical Bayes** is a quick approximation; the full model also
  accounts for uncertainty in `tau`.

## What to do now

1. Run `lesson.py` and look at `output/shrinkage.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 006 changes gear completely, from "how big is it?" to "what
   happens if we act?"
