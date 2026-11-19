# Lesson 009 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_009
cp lessons/phase-1-bayesian-and-causal/lesson-009-quasi-experiments-when-you-cant-randomise/lesson.py my_work_009/
cd my_work_009
python lesson.py
```

---

## 1. DiD by hand

A bike-share operator raised prices in Mombasa in March and left
Nairobi alone. Average daily rides per station:

| | February | April |
|---|---|---|
| Mombasa | 52 | 47 |
| Nairobi | 70 | 68 |

a) What's the DiD estimate of the price rise's effect?
b) What does each of the two single differences (Mombasa April minus
February; April Mombasa minus Nairobi) say, and why is each misleading?

<details>
<summary>Check yourself</summary>

a) (47 - 52) - (68 - 70) = -5 - (-2) = **-3 rides per station per day**.

b) Mombasa before/after says -5, but Nairobi fell by 2 with no price
change, so some of that drop is the season (or whatever else changed
across both cities). April Mombasa vs Nairobi says -21, but Mombasa was
already 18 lower in February; that's a city difference, not the price.

</details>

## 2. Which assumption breaks?

For each story, say whether parallel trends is plausible, and if not,
which way the DiD estimate would be biased.

a) The Mombasa price rise above, but a long matatu strike hit Nairobi
in April only, pushing extra riders onto bikes there.
b) Lanes went to stations that the city picked by drawing names from a
hat among all 60.
c) A loyalty scheme was offered to the stations whose rides had dipped
unusually low last month.

<details>
<summary>Check yourself</summary>

a) **Broken.** A shock that hits only the control group in the after
period. Without the strike Nairobi would have fallen more than 2, so the
control "change" is too high and DiD makes the price rise look **worse**
than it was (more negative).

b) **Fine**, and better than fine: that's a randomised experiment, and
DiD just tightens the estimate by removing station levels.

c) **Broken, in a sneaky way.** Stations chosen *because* of a
temporary dip tend to bounce back by themselves (regression to the
mean, also known as the "Ashenfelter dip"). DiD credits the bounce to
the scheme, so it's biased **upwards**. An event study shows it as a
dip in the last pre-period weeks.

</details>

## 3. Why cluster?

In one sentence each: (a) why are plain OLS standard errors too small
in section 3? (b) why cluster by *station* and not by *week*?

<details>
<summary>Check yourself</summary>

a) A station's weekly rides are correlated over time (this week's luck
carries into next week), so 26 weeks from one station carry much less
independent information than OLS assumes.

b) Treatment was assigned to stations, and it's the station-level
errors that persist across weeks; the rule of thumb is to cluster at
the level the treatment was assigned (and at least at the level of the
correlation).

</details>

## 4. Hands-on: how sticky is the noise?

In your copy, find the line in `world()` with `0.6 * noise[:, w - 1]`.
That 0.6 is how much of last week's luck carries into this week. Change
it to `0.0`, run, and look at the two lines in section 3: the SEs and
the coverage. Then try `0.9`.

Predict first: with no stickiness, which SE is right?

<details>
<summary>Check yourself</summary>

With `0.0`, the weeks really are independent, and the two methods agree:

```
  SE, plain OLS / clustered by station           -> 0.43 / 0.36
  95% CI covers the truth: plain / clustered     -> 95.6% / 95.4%
```

With `0.9`, plain OLS falls apart:

```
  SE, plain OLS / clustered by station           -> 0.65 / 1.70
  95% CI covers the truth: plain / clustered     -> 48.6% / 91.2%
```

The plain "95%" interval is right less than half the time. Real weekly
and daily series are usually sticky like this, which is why clustering
is the default, not a refinement.

</details>

## 5. Hands-on: too few controls

Set the original `0.6` back. Change the top line to
`S, W, T0, N_TREAT = 60, 26, 16, 55`, so only 5 stations are left
without lanes. Run and check section 3 again.

<details>
<summary>Check yourself</summary>

```
  SE, plain OLS / clustered by station           -> 0.85 / 1.25
  95% CI (clustered)                             -> [+3.75, +8.65]
  95% CI covers the truth: plain / clustered     -> 73.2% / 86.8%
```

The interval is much wider, which is honest: 5 control stations can
only tell you so much about "what else changed". But even the clustered
interval now covers the truth only 87% of the time, because the
sandwich formula needs a fair number of clusters on *both* sides to
work well. With a handful of treated or control units, use methods
built for that (a wild cluster bootstrap, a permutation test over which
units were treated, or synthetic control), and treat the result with
humility.

</details>

## 6. Optional puzzle: the placebo outcome

A placebo **date** checks the before period. There's another kind of
placebo: an **outcome** the treatment can't plausibly affect. In
`lesson.py`'s world, suggest one placebo outcome you'd look at for the
bike lanes, and say what result would make you worried.

Then, for a challenge: write a function `placebo_grid(y, treated)`
that runs the placebo DiD for every fake start week from 4 to 12 and
prints them. Run it on the good world and on `world(trend_gap=0.5)`.
What pattern separates the two?

<details>
<summary>Check yourself</summary>

A good placebo outcome: **night-time rides** (if the lanes only change
daytime commuting), or rides at **stations two kilometres away** from
any lane. If DiD "found" an effect on those, something other than the
lanes is moving the treated stations.

```python
def placebo_grid(y, treated):
    for k in range(4, 13):
        print(k, f"{placebo(y, treated, fake_t0=k):+.2f}")
```

In the good world they all stay within about 1.6 rides of zero
(between -1.59 and +0.03). In the growing-stations world every one is
between +4.6 and +6.5, because every fake start date sees later weeks
higher than earlier ones. One placebo can be unlucky; a whole row of them pointing
the same way is a trend.

</details>
