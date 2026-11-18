# Lesson 008 - Confounding, graphs and what to adjust for

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 4 | Wednesday 2026-11-18

> **Goal:** Draw a small causal graph, name a confounder, and adjust
> without opening a collider.

Time: about 50 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 007 had the luxury of a coin. Often you don't get one. Nobody
will randomly assign riders to e-bikes for a month just so we can
measure whether e-bikes make trips shorter. We have last month's rides,
and riders chose their bikes themselves.

They chose for reasons. Long trips and hilly stations push people
towards e-bikes, and long trips and hills also make rides take longer.
So in the raw data e-bike rides might well look *slower*. The standard
response is "control for things", meaning throw variables into a
regression. That's half right. Some variables fix the comparison, some
leave it untouched, and some quietly **break** it. Today you'll learn to
tell which is which *before* you look at the estimates, by drawing a
picture.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-008-confounding-graphs-and-what-to-adjust-for/lesson.py
```

It runs in about a second and saves `output/adjustment.png` next to the
script: the graph on the left, every estimate on the right.

## 1. Draw the graph

A **causal graph** (or DAG, for directed acyclic graph) has one node per
variable and an arrow wherever we believe one thing directly affects
another. In code it's just a dictionary of parents:

```
    ebike     <- dist, hill, promo
    speed     <- ebike, hill
    duration  <- dist, hill, speed, rain
    flagged   <- ebike, duration
```

In words: trip distance, a hilly station and an in-app discount
(`promo`) all make an e-bike more likely. The e-bike makes you faster,
hills make you slower, and speed, distance, hills and rain set how long
the trip takes. Finally, rides get **flagged** for review more often when
they're on an e-bike (battery checks) or run long.

The graph is an assumption, not a result. Its power is that it's
**explicit**: anyone can point at an arrow and argue with it. And the
arrows you *leave out* are the strongest claims; there's no arrow from
rain to e-bike, so we're claiming rain doesn't change bike choice.

Next, list every path between `ebike` and `duration`, ignoring arrow
direction:

```
    ebike <- dist -> duration                    non-causal open with no adjustment: True
    ebike <- hill -> speed -> duration           non-causal open with no adjustment: True
    ebike <- hill -> duration                    non-causal open with no adjustment: True
    ebike -> speed <- hill -> duration           non-causal open with no adjustment: False
    ebike -> speed -> duration                   causal     open with no adjustment: True
    ebike -> flagged <- duration                 non-causal open with no adjustment: False
```

Only one path is the effect we want: `ebike -> speed -> duration`. Every
other path is a way for `ebike` and `duration` to be associated
**without** the bike causing anything. Three rules decide whether a
path carries association:

- **Chain** `A -> M -> B` and **fork** `A <- M -> B`: open, unless you
  adjust for `M`, which blocks it.
- **Collider** `A -> M <- B`: *blocked* by default, and adjusting for
  `M` (or anything `M` causes) **opens** it.

That's the whole of **d-separation**, and `is_open()` in `lesson.py`
is those rules in a dozen lines. Paths that start with an arrow *into*
the treatment (`ebike <- ...`) are **backdoor paths**; they're where
confounding lives.

## 2. Simulate the world the graph describes

We generate 20,000 rides from exactly that graph. An e-bike adds 8 km/h
and every km/h saves half a minute, so the true effect is **-4 minutes**.

```
  share of rides on an e-bike                    -> 69.9%
  true effect of an e-bike on duration           -> -4.0 min
  mean dist: e-bike / pedal                      -> 3.36 / 2.15 km
  share on a hill: e-bike / pedal                -> 0.35 / 0.17
```

E-bike trips are 1.2 km longer and twice as likely to start on a hill.
Those are the backdoors, already visible in the data.

## 3. Close the backdoors

We regress duration on `ebike` plus an adjustment set, and for each set
the code applies the **backdoor criterion**: a set is valid if it
contains nothing caused by the treatment, and it blocks every backdoor
path.

```
  adjust for {}                                  -> +1.53 (SE 0.147)   valid? no: open ebike <- dist -> duration
  adjust for {dist}                              -> -3.24 (SE 0.082)   valid? no: open ebike <- hill -> speed -> duration
  adjust for {dist, hill}                        -> -4.04 (SE 0.079)   valid? yes
```

The naive comparison says e-bikes make trips **1.5 minutes longer**.
That's not noise (look at the standard error); it's **confounding**.
`dist` is a **confounder**: a common cause of the treatment and the
outcome. Adjusting for it gets us most of the way, to -3.2. But two
backdoors still run through `hill`. Add it and we land on -4.04, right on
the truth.

Notice that the graph told us `{dist, hill}` was enough before we
looked at a single estimate. That's the point: **decide the adjustment
set from the graph, not by trying sets until the number looks
sensible.**

## 4. Don't adjust for the mechanism

It's tempting to "control for everything we have". Add `speed`:

```
  adjust for {dist, hill, speed}                 -> +0.08 (SE 0.196)   valid? no: speed is downstream of ebike
```

The effect vanishes. `speed` is a **mediator**: it's *how* the e-bike
works. Holding speed fixed asks "what does an e-bike do for a rider
going the same speed?", and the answer is nothing, because the bike's
whole effect runs through speed:

```
  causal paths open, adjusting {dist, hill}      -> ['ebike -> speed -> duration']
  causal paths open, adjusting {dist, hill, speed} -> none
```

Sometimes the **direct** effect (the part not through a mediator) is
what you want, but you should choose that on purpose. For the total
effect, never adjust for anything the treatment causes. A common real
version: measuring whether a training course raised salaries while
"controlling for job title", when getting promoted is how the course
pays off.

## 5. Don't select on a consequence

Flagged rides are a natural thing to analyse; they're the ones somebody
looked at closely. About half of rides are flagged:

```
  share of rides flagged                         -> 52.6%
  {dist, hill}, flagged rides only               -> -5.37 (SE 0.133)
  adjust for {dist, hill, flagged}               -> -5.29 (SE 0.075)   valid? no: flagged is downstream of ebike
```

Both ways of using `flagged`, filtering on it or adding it to the
regression, push the estimate to about -5.3, a third too big. `flagged`
is a **collider** on the path `ebike -> flagged <- duration`:

```
  ebike -> flagged <- duration open, Z={}        -> False
  ebike -> flagged <- duration open, Z={flagged} -> True
```

Left alone, a collider blocks its path. Condition on it and it opens.
The intuition: among flagged rides, a pedal-bike ride must have been
flagged for being *long* (it had no battery to check), while an e-bike
ride could have been flagged for its battery even if it was quick. So
inside the flagged group, pedal bikes look extra slow and e-bikes look
extra fast. A spurious association appears out of pure selection.

Colliders hide in innocent places: "only riders who left a rating",
"only loans that were approved", "only patients who were admitted".
Any filter that depends on both treatment and outcome can manufacture
an effect, or hide one.

## 6. Variables that change the error bar, not the answer

Two more variables are safe to include, but they're not equal:

```
  adjust for {dist, hill, rain}                  -> -3.93 (SE 0.066)   valid? yes
  adjust for {dist, hill, promo}                 -> -4.03 (SE 0.091)   valid? yes
  SE with rain vs without                        -> 0.066 vs 0.079 (-16%)
  SE with promo vs without                       -> 0.091 vs 0.079 (+16%)
```

- `rain` affects only the outcome. Including it soaks up noise in
  duration, so the standard error **falls**, just like the
  pre-treatment covariate in lesson 007. These are good to include.
- `promo` affects only the treatment. Including it removes some of the
  variation in who rides an e-bike, which is exactly the variation we
  learn from, so the standard error **rises**. No bias, just a blurrier
  estimate. Leave such variables out. (They have a better use: lesson
  009 and later lessons exploit "nudges" like this as natural
  experiments.)

## What the graph can't do

Everything here worked because we *recorded* `dist` and `hill`. Suppose
the real driver of bike choice were "running late for work", which also
makes people ride faster and which nobody logs. No adjustment set made
of observed variables would close that backdoor, and no amount of data
would warn you. That's **unmeasured confounding**, and it's why
randomised experiments stay the gold standard. When you can't randomise
and can't measure the confounder, you need a design that sidesteps it,
which is lesson 009.

## What to remember

- Write your assumptions as a **causal graph**; the missing arrows are
  the strong claims.
- A **confounder** is a common cause of treatment and outcome. Adjust
  for enough of them to **block every backdoor path**.
- Never adjust for a **mediator** if you want the total effect, or for
  anything else **downstream** of the treatment.
- Conditioning on a **collider**, by adjusting or by filtering, opens a
  path and creates bias out of nothing.
- Outcome-only variables shrink the error bar; treatment-only variables
  inflate it.
- **Pick the adjustment set from the graph first**, then estimate.

## What to do now

1. Run `lesson.py` and compare the graph with the estimates in
   `output/adjustment.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 009 tackles the case where the confounder isn't in your data:
   difference-in-differences on a before/after series with a control.
