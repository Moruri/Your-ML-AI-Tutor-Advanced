# Lesson 007 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_007
cp lessons/phase-1-bayesian-and-causal/lesson-007-randomised-experiments-and-ab-tests/lesson.py my_work_007/
cd my_work_007
python lesson.py
```

---

## 1. Size it on paper

A station manager wants to test a "return by 8am, get 10 bob off"
offer. The late rate at her stations is 0.15 and she only cares about a
drop of 2 points or more. Using the formula in section 1, roughly how
many riders per arm does she need? What if she'd settle for detecting 4
points?

<details>
<summary>Check yourself</summary>

`(1.96 + 0.84)^2 * 2 * 0.15 * 0.85 / 0.02^2` is about `7.84 * 0.255 /
0.0004`, roughly **5,000 per arm**. For 4 points it's a quarter of that,
about **1,250 per arm**. (`n_per_arm(0.15, 0.02)` gives 5,004 and
`n_per_arm(0.15, 0.04)` gives 1,251.) If her stations only see 600
riders a week, the 2-point version takes four months, and that's worth
knowing *before* she starts.

</details>

## 2. What does the interval say?

A colleague reports: "Text vs control, ATE -0.012, 95% CI [-0.031,
+0.007], p = 0.21. The text doesn't work." Is that the right
conclusion? What would you write instead?

<details>
<summary>Check yourself</summary>

No. The interval includes zero, but it also includes -0.03, an effect
the business would care about. The honest summary is "**inconclusive**:
the data are consistent with no effect and with a worthwhile one". The
experiment was probably underpowered. "Not significant" means "not
shown", not "shown to be zero". If you want to claim the effect is too
small to matter, you need an interval that sits entirely inside the
too-small zone, here above -0.03.

</details>

## 3. Pick the right unit

Operations proposes randomising **stations** instead of riders: half
the stations send texts, half don't. Give one good reason to do that,
and say what it does to the sample size.

<details>
<summary>Check yourself</summary>

Good reasons: riders at the same station talk to each other, or the
text changes how bikes are rebalanced across the station, so treating
one rider affects untreated neighbours (**interference**). Randomising
whole stations keeps the arms separate. The cost: riders at one station
are alike, so the **effective sample size** is much closer to the
number of stations than the number of riders. With 12 stations you'd
have very little power. The analysis must also respect the clusters
(for example, compare station-level late rates), or the interval will be
far too narrow.

</details>

## 4. Hands-on: an underpowered rerun, and a fixed one

In your copy:

1. Change section 2 to use `N = 400` instead of the computed sample
   size, rerun, and note the interval from section 3. Does it still
   meet the decision rule?
2. Back with the full `N`, find the smallest `n` per arm for which the
   simulated power in section 5 reaches 80% for a true effect of
   -0.03. Try `many(n, scale=0.03 / -TRUE_ATE)` for a few values of `n`
   between 2,000 and 3,036.

Predict first: will the simulation agree with the formula's 3,036?

<details>
<summary>Check yourself</summary>

1. With 400 per arm the standard error rises to about 0.028, so the
   interval is roughly +/-0.055 wide. With the lesson's seed you get
   about [-0.119, -0.011]: it does exclude zero, but it spans "barely
   anything" to "huge" and the point estimate (-0.065) is inflated, the
   winner's curse from section 5 in a single run. It meets the letter
   of the decision rule while telling you almost nothing about whether
   the effect clears the 3-point MDE.
2. Somewhere around **2,800 to 2,900** per arm gets you to 80% in the
   simulation (each run wobbles by a point or so of power), a bit under
   the formula's 3,036. The formula uses the
   control's variance `p(1 - p)` for both arms, and the text arm has a
   lower late rate and so less variance, so the formula is slightly
   conservative. That's the right way round to be wrong.

</details>

## 5. Optional puzzle: peek without paying

Modify section 6 so that you still look every day, but declare a win
only if `|z| > c`. Find the `c` that brings the 20-look false positive
rate back to about 5%.

<details>
<summary>Check yourself</summary>

Replace `1.96` with a larger threshold and rerun: around **c = 2.7**
gets the 20-look rate near 5% (2.6 gives about 6.5%, 2.8 about 4%). (A single look needs only 1.96.) The
price is power: you need a bigger effect to cross 2.8 at the end. This
is the idea behind group-sequential designs (Pocock-style boundaries
use a constant threshold like this; O'Brien-Fleming-style ones are very
strict early and close to 1.96 at the end, so they cost almost nothing
if you run to completion).

</details>
