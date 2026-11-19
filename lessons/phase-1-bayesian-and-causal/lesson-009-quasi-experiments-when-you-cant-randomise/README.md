# Lesson 009 - Quasi-experiments when you can't randomise

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 5 | Thursday 2026-11-19

> **Goal:** Estimate a treatment effect with difference-in-differences
> on a before/after series with a control.

Time: about 50 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 008 ended on an uncomfortable note. If whatever drives the
treatment isn't in your data, no adjustment set can fix the comparison,
and the data won't warn you. Today we get around that with a **design**
instead of a better model.

Here's the situation. The city painted protected bike lanes past 20 of
our 60 stations. Nobody asked us, nobody flipped a coin, and the city
chose busy stations on main roads, for reasons we never recorded. Did
the lanes bring more rides? We can't randomise and we can't measure the
city's reasoning. What we do have is time: weekly rides at every station
from 16 weeks before the lanes opened to 10 weeks after. That's enough
for one of the most useful tools in applied causal work,
**difference-in-differences** (DiD).

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-009-quasi-experiments-when-you-cant-randomise/lesson.py
```

It runs in a few seconds and saves `output/did.png` next to the script:
the raw series on the left, two event studies on the right.

## 1. Two tempting comparisons, both wrong

The simulated world: 60 stations over 26 weeks. Lanes open at the start
of week 16 next to 20 of them. Lane stations were busier to begin with
(about 25 extra rides a week, never written down anywhere). Every
station shares the same weekly ups and downs (season, plus a long rainy
spell right after the lanes opened), and each station's own luck is
sticky from one week to the next. The lanes truly add **6 rides per
station per week**.

```
  mean weekly rides, lane stations: pre / post   -> 106.7 / 97.6
  mean weekly rides, other stations: pre / post  -> 81.1 / 65.5
  after only: lanes minus no lanes               -> +32.1  (true +6.0)
  lane stations only: after minus before         -> -9.1
```

Two natural comparisons, two wrong answers:

- **After only, lanes vs no lanes:** +32. That's mostly the 25-ride head
  start the busy stations always had. This is lesson 008's confounding:
  the city chose by something that also drives rides.
- **Lane stations, after vs before:** -9. The lanes made things *worse*?
  No: the rain arrived when the lanes did. A before/after comparison
  can't tell the treatment from anything else that changed at the same
  time.

Each comparison has one problem the other doesn't have. That's the clue.

## 2. Difference-in-differences

Use the stations without lanes to measure "what else changed", and
subtract it:

```
  change at lane stations                        -> -9.11
  change at other stations                       -> -15.63
  DiD estimate                                   -> +6.52  (true +6.0)
```

Lane stations dropped by 9 rides. Stations without lanes dropped by 16
over the same weeks, because of the rain. So lane stations did about
**6.5 rides better than they would have**. In one line:

```
DiD = (treated after - treated before) - (control after - control before)
```

The first difference removes anything **fixed about a station**,
including that unrecorded head start. The second difference removes
anything that **hits every station in the same week**, like rain. What's
left is the effect, *if* one assumption holds:

> **Parallel trends:** without the lanes, lane stations would have
> changed by the same amount as the others.

Not the same *level*; DiD is fine with lane stations being busier. The
same *change*. The dashed line on the left of the figure is exactly
this assumption drawn out: the control stations' path, shifted up to
where lane stations started. The effect is the gap between the solid
blue line and that dashed line.

Compare that with what we needed in lesson 008: every confounder
measured. DiD needs something different, that the unmeasured stuff is
either fixed over time or shared across groups. Sometimes that's much
easier to believe. Sometimes it isn't (section 5).

## 3. The same thing as a regression, and an honest error bar

The four-means version gives a number but no error bar. The standard
way to get one is a regression with a dummy for every station, a dummy
for every week, and one more variable, `lanes_open` (1 for a lane
station in week 16 or later):

```
  regression coefficient on lanes_open           -> +6.52
  SE, plain OLS / clustered by station           -> 0.50 / 0.75
  95% CI (clustered)                             -> [+5.04, +8.00]
```

Same estimate, as it must be with a balanced panel. The station dummies
are the first difference, the week dummies the second. The regression
form earns its keep later: it handles extra covariates, unbalanced data
and many time periods without changes.

Now look at the two standard errors. Plain OLS treats all 1,560
station-weeks as independent pieces of evidence. They aren't. A station
that was busy last week tends to be busy this week, so 26 weeks of one
station are worth much less than 26 independent stations. **Clustered**
standard errors allow any correlation within a station and only treat
*stations* as independent. `ols()` in `lesson.py` builds them in a few
lines (the "sandwich" formula). Who's right? Re-run the whole world 500
times:

```
  mean DiD estimate over 500 worlds              -> +6.03
  95% CI covers the truth: plain / clustered     -> 70.6% / 92.6%
```

The estimate is unbiased either way. But the plain "95%" interval
misses the truth almost a third of the time. Clustered gets close to
95%; it's a little short because 60 clusters, 20 of them treated, isn't
a huge number. This is one of the most common mistakes in published
DiD work: **cluster by the unit that was treated**.

## 4. Check the assumption you can check

You can never see the "no lanes" version of the after period. But you
can look at the **before** period, where both groups were untreated and
should therefore move together. Two checks:

**Event study.** Plot the treated-minus-control gap week by week,
relative to the last week before the lanes (week 15):

```
  pre-period gaps (weeks 0-14), mean |gap|       -> 0.95
  post-period gaps (weeks 16-25), mean           -> +6.47
```

Before the lanes, the gap wobbles around zero (the blue line on the
right of the figure). Then it jumps by about 6 and stays there. That
picture is the most convincing thing you can show anyone about a DiD.

**Placebo.** Pretend the lanes opened at week 8, and run DiD using only
weeks 0-15, before anything really happened:

```
  placebo DiD, fake start at week 8              -> -0.16
```

About zero, as it should be. A placebo that "finds" an effect where
none can exist tells you the design is picking up something else.

## 5. When parallel trends fail

Now change one thing. Suppose the city put lanes where new flats were
going up, so lane stations were **already growing**, half a ride per
week per week, lanes or no lanes. The true effect is still +6:

```
  DiD estimate (true effect still +6.0)          -> +14.39
  pre-period gap: week 0 vs week 14              -> -10.15 vs +0.11
  placebo DiD, fake start at week 8              -> +5.78
```

DiD more than doubles the effect, because it credits the lanes with the
growth that would have happened anyway. But this time the checks
complain. The orange event study climbs steadily through the whole
before period, and the placebo finds a "+5.8 effect" from a fake start
date. If you'd skipped section 4, you'd never know.

How bad can it get? The same growing stations with **no** lane effect
at all:

```
  no-effect world, plain DiD                     -> +7.34
```

A confident, entirely fictional finding.

One common repair is to fit the pre-period trend of the gap and project
it forward:

```
  pre-trend slope of the gap (rides/week/week)   -> +0.70
  DiD after projecting the pre-trend forward     -> +5.34
```

Much closer to 6. But be honest about what you did: you swapped one
assumption (parallel trends) for another (the extra growth would have
continued in a straight line), and the slope itself is noisy (0.70
estimated against a true 0.5). Present it as a sensitivity check, not a
fix.

## Where DiD shows up

- A price change in one city but not another.
- A new feature rolled out to Android before iOS.
- A policy (a minimum wage, a congestion charge) that applies on one
  side of a border.
- A model launched in some regions first because of a staged rollout.

The pattern is always the same: a group that changed, a group that
didn't, and data from before and after for both. If you control the
rollout, you can do better still by randomising which regions go first,
which turns a quasi-experiment back into a real one (lesson 007).

Other designs in the same family, worth knowing by name: **regression
discontinuity** (treatment switches on at a cutoff, like a credit score
of 600, so compare people just either side), **instrumental variables**
(a nudge, like lesson 008's `promo`, that shifts treatment but affects
the outcome only through it) and **synthetic control** (build the
control group as a weighted mix of untreated units that tracks the
treated one before the change). Each swaps "measure every confounder"
for a different, checkable-ish assumption.

## What to remember

- Before/after alone confuses the treatment with **anything else that
  changed then**. Treated vs control alone confuses it with **how the
  groups always differed**. DiD subtracts both.
- The key assumption is **parallel trends**: same change, not same
  level, in the absence of treatment.
- You can't test it in the after period, but you can **check the
  before period** with an event study and a placebo date. Always do.
- **Cluster standard errors by the treated unit.** Weekly data from one
  station isn't independent, and plain OLS intervals can be far too
  narrow.
- Correcting for a pre-trend trades one assumption for another; report
  it as a sensitivity check.

## What to do now

1. Run `lesson.py` and study both panels of `output/did.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 010 is the phase 1 mini-project: take a simulated experiment
   all the way from raw data to a recommendation a product lead could
   act on, Bayesian-style and frequentist-style.
