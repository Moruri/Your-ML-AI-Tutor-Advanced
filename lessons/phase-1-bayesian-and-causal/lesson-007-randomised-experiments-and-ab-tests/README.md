# Lesson 007 - Randomised experiments and A/B tests

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 4 | Wednesday 2026-11-18

> **Goal:** Design a simple experiment, estimate an average treatment
> effect, and report it with a confidence interval.

Time: about 50 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 006 ended on an uncomfortable note. Last month's reminder texts
went to riders operations judged risky, so comparing texted and untexted
riders made the texts look *harmful*. The data couldn't separate what
the text did from who got it.

The cure is almost embarrassingly simple: **let a coin decide**. If
chance alone picks who gets a text, the two groups differ only by
chance and by the text, and chance is something statistics knows how
to measure. That's a randomised experiment, and in product teams it's
called an **A/B test**. Simple to describe, easy to get subtly wrong.
Today we do one properly, from the sample-size sum on paper to the
sentence you'd write for the product lead.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-007-randomised-experiments-and-ab-tests/lesson.py
```

It takes a few seconds and saves `output/experiments.png` next to the
script. The riders are the same simulated world as lesson 006, so we
know the true effect of a text on a typical rider: **-0.053**, about
five points off the late rate.

## 1. Design before data

Write these down before a single text goes out:

- **Question:** does an evening-before text reduce late returns?
- **Unit:** a rider. Each rider is randomised once and stays in their arm.
- **Metric:** late on their next return (yes/no), measured the same way in both arms.
- **Minimum detectable effect (MDE):** the smallest effect worth acting
  on. Texts cost money, and the product lead says anything under
  **3 points** isn't worth it.
- **Decision rule:** roll out if the 95% confidence interval sits
  entirely below zero.

The MDE drives the sample size. For two proportions with base rate `p`,
5% significance and 80% power:

```
n per arm = (z_0.975 + z_0.8)^2 * 2 p (1 - p) / MDE^2
```

```
  base late rate (no text)                       -> 0.224
    riders per arm to detect -0.05               -> 1,093
    riders per arm to detect -0.03               -> 3,036
    riders per arm to detect -0.02               -> 6,830
    riders per arm to detect -0.01               -> 27,317
```

Halve the effect you want to see and you need **four times** the
riders, because the noise in an average shrinks only with the square
root of `n`. We go with 3,036 per arm.

**Power** is the chance the test comes out significant *if* the true
effect is at least the MDE. 80% means one in five worthwhile effects
would still slip past. Choosing it up front is what stops "we'll run it
until it looks interesting".

## 2. Randomise, then check the coin

We shuffle exactly 3,036 "text" labels and 3,036 "control" labels
across 6,072 riders. Then we check the groups look alike on things
measured *before* the experiment:

```
  mean past_late: text / control                 -> 1.67 / 1.67
  SMD past_late, SMD dist                        -> +0.000, -0.019
```

The **standardised mean difference** (SMD) is the gap between arms in
units of standard deviation; under about 0.1 is fine. This balance
check is not a test of whether randomisation "worked" (a fair coin
always works). It catches **bugs**: a broken assignment script, an arm
that silently dropped Android users, a filter applied to one side only.
Run it every time.

## 3. The estimate and its interval

The **average treatment effect** estimate is simply the difference in
late rates, and its standard error comes from the two arms' variances:

```
SE = sqrt( p_text (1 - p_text) / n_text  +  p_ctrl (1 - p_ctrl) / n_ctrl )
```

```
  late rate: text / control                      -> 0.173 / 0.232
  ATE estimate (text minus control)              -> -0.0586  (true -0.0527)
  standard error                                 -> 0.0103
  95% confidence interval                        -> [-0.0788, -0.0385]
  p-value (two-sided z-test)                     -> 1.2e-08
```

The interval is the headline, not the p-value. It says the text cuts
late returns by somewhere between about 4 and 8 points, comfortably
below zero and comfortably past the 3-point MDE. Our point estimate
(-0.059) misses the truth (-0.053) by a little, exactly as an estimate
should; the interval covers it.

A second check uses nothing but the coin. If the text did nothing, the
labels are arbitrary, so shuffle them and recompute the gap:

```
  shuffles with a gap at least this big          -> 0 of 5,000
  largest gap chance produced                    -> 0.0389
```

This **permutation test** makes no assumptions about normality. Chance
alone never produced a gap near 0.059 in 5,000 tries. The formula and
the shuffle agree.

## 4. A tighter interval for free

We know each rider's past late rides from *before* the coin flip. Since
it predicts lateness strongly, we can regress the outcome on the
treatment plus that covariate (centred) and distance:

```
  adjusted ATE estimate                          -> -0.0576
  standard error: plain / adjusted               -> 0.0103 / 0.0095
  like running the plain test with               -> 1.17x the riders
```

Same experiment, a smaller standard error, worth 17% more riders. This
is the idea behind **CUPED** and similar tricks used by big experiment
platforms. The golden rule: only adjust for things measured **before**
treatment. Adjusting for something the text could have changed breaks
the comparison, which is the heart of lesson 008.

## 5. Run the experiment 2,000 times

One experiment is one draw. Repeating the whole thing in simulation
shows what the method promises:

```
  n=3,036/arm: CI covers the truth               -> 95.0%
  n=3,036/arm: power (p < 0.05)                  -> 100.0%
  n=3,036/arm: mean estimate when significant    -> -0.0526
  n=3,036/arm, true effect -0.03: power          -> 83.2%
```

The 95% interval covers the true effect 95% of the time; that's what
"95%" means. It's a statement about the **procedure**, not about one
interval. The real effect (-0.053) is bigger than our MDE, so we detect
it every time. Shrink it to exactly -0.03 and power drops to about 83%,
a touch above the 80% we planned for because the formula plays safe
(the text arm is less variable than the control).

Now the underpowered version, 400 riders per arm:

```
  n=400/arm: power (p < 0.05)                    -> 46.4%
  n=400/arm: mean estimate when significant      -> -0.0770
  n=400/arm: exaggeration when significant       -> 1.5x
```

Small experiments don't just miss real effects more often. **The ones
that do come out significant overstate the effect**, here by about
half, because only the lucky draws clear the bar. This is the
**winner's curse**, and it's why a small "huge win" so often shrinks
when it's rolled out.

Open `output/experiments.png`. On the left, 50 of the experiments: each
line is a 95% interval, and the red ones missed the truth. About 1 in
20 should.

## 6. Peeking

Experiments run for days, and dashboards update live. The temptation:
check every morning and stop the moment p < 0.05. To see the cost, we
run **A/A tests**, where both arms get nothing, so every "significant"
result is a false alarm. 20 days, 150 riders per arm per day:

```
     1 look(s): false positive rate              -> 4.9%
     2 look(s): false positive rate              -> 8.9%
     5 look(s): false positive rate              -> 15.7%
    10 look(s): false positive rate              -> 20.4%
    20 look(s): false positive rate              -> 26.2%
```

Looking once gives the promised 5%. Looking daily and stopping on the
first good day gives **26%**: one in four do-nothing changes "wins". The
right panel of the figure shows the climb. Each look is another lottery
ticket, and stopping on a win keeps only the winners.

The fixes: fix the sample size in advance and look once, or use a
method built for repeated looks (sequential tests with adjusted
thresholds, or the Bayesian view from lessons 002-004, where you report
a posterior instead of a reject/accept decision). Looking at the
dashboard for *bugs* is fine. Deciding from it early is not.

## What to remember

- Write down **question, unit, metric, MDE and decision rule** before
  you start. The MDE sets the sample size, and halving it quadruples `n`.
- **Randomise**, then run a **balance check** on pre-treatment
  variables to catch bugs.
- Report the **ATE with a confidence interval**. The interval is the
  decision; the p-value is a footnote.
- Adjusting for **pre-treatment** covariates tightens the interval for
  free.
- Underpowered experiments miss real effects and **exaggerate** the
  ones they find.
- **Peeking** and stopping early inflates false positives far beyond 5%.

## What to do now

1. Run `lesson.py` and look at both panels of `output/experiments.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 008 asks what to do when you *can't* randomise: draw the
   causal graph, find the confounders, and adjust for the right things
   (and not the wrong ones).
