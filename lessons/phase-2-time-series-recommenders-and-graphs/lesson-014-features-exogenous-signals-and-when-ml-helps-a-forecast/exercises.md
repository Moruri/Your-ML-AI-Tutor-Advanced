# Lesson 014 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_014
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-014-features-exogenous-signals-and-when-ml-helps-a-forecast/lesson.py my_work_014/
cd my_work_014
python lesson.py
```

---

## 1. Known or not?

On Monday morning you forecast **Saturday** (lead 5). Which of these
columns can go in the row, as written?

a) Rides last Saturday.
b) Rides on Wednesday this week.
c) The average of the 7 days ending the day before the target.
d) Whether Saturday is a public holiday.
e) Saturday's rainfall from the weather station.
f) Saturday's rainfall from Monday's weather forecast.

<details>
<summary>Check yourself</summary>

a) **Yes**: 7 days before the target, and 7 >= 5.

b) **No**: Wednesday is in the future on Monday morning.

c) **No**: it ends on Friday, and includes four days you haven't seen.
A trailing average is fine only if it ends at least 5 days before the
target, like `lag7_13_mean`.

d) **Yes**: the calendar is known.

e) **No**: that's the oracle from section 4.

f) **Yes**: that's exactly what `rain_fc` at lead 5 is. It only works
if you stored Monday's forecast instead of overwriting it.

</details>

## 2. The tree's ceiling

A tree model is trained on raw (not detrended) rides from a year in which
daily rides ranged from 700 to 1,350. The trend says next month should
average about 1,400 on weekdays.

a) What's the most the model can possibly forecast for any day?
b) Which way will its errors lean, and which row of section 3 shows it?
c) Name one fix.

<details>
<summary>Check yourself</summary>

a) **1,350**, and in practice less. Every leaf predicts an average of
training targets, so no forecast can exceed the largest one seen.

b) **Under-forecasts**: a negative bias. That's "boosted trees, raw
rides", with a bias of **-26.4** against -5.8 once detrended.

c) Model the rides **minus a trend** fitted on the past (what the lesson
does), or model the **change** from a recent level (for example, rides
minus `lag7`), or use a linear model, which can extrapolate.

</details>

## 3. What's a better weather forecast worth?

From section 3: trees without rain 57.5, with the real forecast 45.5,
with the actual rain 37.9.

a) What share of the possible gain from rain does the real forecast
capture?
b) A vendor offers a forecast that would close half of the remaining
gap. Roughly what MAE should you expect, and what would you check
before you believe them?

<details>
<summary>Check yourself</summary>

a) Possible gain: 57.5 - 37.9 = 19.6. Captured: 57.5 - 45.5 = 12.0.
So about **61%**.

b) Half of the remaining 7.6 is 3.8, so about **41.7**. Before
believing it, check that their backtest used forecasts **as issued**
on each day (an archive of past forecasts), not today's re-analysis of
past weather, and run it through your own rolling origin.

</details>

## 4. Hands-on: features beat algorithms

The trees won on wet days because they learned the shape of the rain
effect. Give the linear model that shape yourself. In `onehot()`, add
three columns, using `rain_fc` and `match` from `X`:

- `1 - exp(-rain_fc / 8)`: rain whose effect levels off,
- the same, times a weekend flag (`dow >= 5`),
- `match` times a dry flag (`rain_fc < 2`).

Re-run section 3. How does the linear model compare with the trees now?

<details>
<summary>Check yourself</summary>

```python
    rain = X[:, names.index("rain_fc")]
    r = 1 - np.exp(-rain / 8)
    extra = [r, r * (dow >= 5), X[:, names.index("match")] * (rain < 2)]
    return np.column_stack([np.ones(len(X)), X[:, other]]
                           + [(dow == k) for k in range(1, 7)] + extra).astype(float)
```

```
  linear, all features                           -> MAE  41.7 | day 1 / 7:  35.7 /  42.6
  boosted trees, detrended                       -> MAE  45.5 | day 1 / 7:  35.2 /  51.9
```

The linear model jumps from 45.6 to **41.7**, now clearly ahead of the
trees. If you know the shape of an effect, writing it down beats
asking a model to discover it from a few hundred days.

A fair warning: we used the simulation's own saturation constant (8 mm),
so this is the best case. In real life you'd guess the shape from
experience or a plot, and get some of this gain. That's still often the
cheapest win available.

</details>

## 5. Hands-on: average them

Lesson 013 found that averaging different kinds of model helps. Add
a model that averages "linear, all features" and "boosted trees,
detrended" (both from the original `onehot()`), and run section 3.

<details>
<summary>Check yourself</summary>

```python
MODELS["average of linear and trees"] = lambda o: (fc_table(o, "linear") + fc_table(o, "trees")) / 2
```

```
  linear, all features                           -> MAE  45.6
  boosted trees, detrended                       -> MAE  45.5
  average of linear and trees                    -> MAE  43.9
```

**43.9**, better than either. The linear model is better on dry days,
the trees on wet days, so their errors partly cancel. When two models
tie overall but win in different places, an average is nearly always
worth trying.

</details>

## 6. Optional puzzle: the copied weekend

The seasonal naive's error on day 7 is **169**, more than twice its day
1 error of 73, even though "last week" is equally far away for both
(the origins are Mondays, so day 1 copies last Monday and day 7 copies
last Sunday). Explain why, using two facts about how this city was
built.

<details>
<summary>Check yourself</summary>

Day 7 is a **Sunday**, and both new effects are concentrated at weekends:

1. **Rain hurts 1.7 times more at weekends.** If last Sunday was wet
and this one is dry (or the other way round), the copy is off by up to
about 340 rides, against about 200 on a weekday.
2. **Matches are mostly at weekends**, and they come and go: copying
last Sunday's match crowd (+180) onto a Sunday without one, or missing
this Sunday's, is a big miss either way.

Copying last week copies last week's weather and fixtures. The models
with `rain_fc` and `match` columns don't have this problem: their day 7
errors are about 50.

</details>
