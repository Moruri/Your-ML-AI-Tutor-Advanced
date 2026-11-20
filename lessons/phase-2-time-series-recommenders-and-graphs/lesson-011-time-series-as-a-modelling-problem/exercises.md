# Lesson 011 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_011
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-011-time-series-as-a-modelling-problem/lesson.py my_work_011/
cd my_work_011
python lesson.py
```

---

## 1. Baselines by hand

One station's rides over two weeks (Monday to Sunday):

| | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| Week 1 | 40 | 44 | 42 | 46 | 50 | 30 | 20 |
| Week 2 | 42 | 46 | 45 | 47 | 52 | 33 | 21 |

a) Forecast next Monday, Wednesday and Sunday with the naive forecast
and with the seasonal naive forecast.
b) Next week turns out to be 44, 47, 46, 49, 53, 34, 22. What's the
MAE of each forecast over the whole week?
c) The in-sample seasonal-naive MAE (week 2 predicted from week 1) is
the MASE scale. Compute it, and the MASE of the seasonal naive
forecast for next week.

<details>
<summary>Check yourself</summary>

a) Naive: 21 for every day (the last value was a Sunday). Seasonal
naive: Monday 42, Wednesday 45, Sunday 21.

b) Naive errors: 23, 26, 25, 28, 32, 13, 1, so MAE = 148 / 7 =
**21.1**. Seasonal naive errors: 2, 1, 1, 2, 1, 1, 1, so MAE = 9 / 7 =
**1.29**.

c) Week 2 minus week 1: 2, 2, 3, 1, 2, 3, 1, so the scale is 14 / 7 =
**2.0**. MASE = 1.29 / 2.0 = **0.64**: better than the seasonal naive
usually does in-sample, which is a sign of a calm week, not a great
model.

</details>

## 2. Which baseline would you use?

For each series, name the baseline you'd set first, and why.

a) Hourly rides at one station, with a strong morning and evening peak.
b) Monthly new rider sign-ups, growing steadily for three years, no
obvious seasonal pattern.
c) Daily battery swaps across the fleet, which depend mostly on how
many rides happened yesterday.

<details>
<summary>Check yourself</summary>

a) **Seasonal naive with a period of 24** (same hour yesterday), or
168 (same hour last week) if weekends look different. A daily rhythm
dominates hourly data.

b) **Drift.** With steady growth and no season, a straight line from
the first to the last value is the natural bar. Naive would always lag
behind the growth.

c) **Naive** (yesterday's value) is the obvious bar, and you might add
a weekly seasonal naive too if swaps follow the weekly ride pattern.
Then any model has to beat both.

</details>

## 3. The missing ends

In one or two sentences: why does the centred 7-day trend have no
value for the last three days, and why does that matter for a
forecasting feature like "trend at day *t*"?

<details>
<summary>Check yourself</summary>

The centred window at day *t* averages days *t-3* to *t+3*, so it needs
three days that haven't happened yet. If you build "trend at day *t*"
on historical data with a centred window and feed it to a model, every
training row has used the future, and at the forecast origin the
feature can't be computed at all. Use a trailing window (days *t-6* to
*t*) for features, and accept that it lags.

</details>

## 4. Hands-on: a longer holdout

In your copy, change `H = 28` to `H = 56`, so the holdout runs from 6
October to 30 November. Before you run it: will the seasonal naive
still win among the simple baselines?

<details>
<summary>Check yourself</summary>

```
  seasonal naive (last week)                     -> MAE  55.0  RMSE  73.5  MASE 1.01
  seasonal mean (last 4 weeks)                   -> MAE  46.5  RMSE  64.4  MASE 0.85
  decomposition (flat trend + index)             -> MAE  51.8  RMSE  71.1  MASE 0.95
```

No. Now the **seasonal mean** wins, and the other two got much worse.
Two things changed: 8 weeks is a long way to copy one week forward, and the
holdout now includes Mashujaa Day (20 October), which no baseline
expects. Averaging four weeks smooths out the luck of any single week.

The real lesson: **the ranking of models depended on which days you
held out.** One holdout is one draw. Lesson 012 replaces it with many.

</details>

## 5. Hands-on: faster growth

Put `H = 28` back. In `rides_series`, change the trend from
`900 + 0.55 * t` to `900 + 2.0 * t`, so the network grows almost four
times faster. Which baseline suffers most, and which benefits?

<details>
<summary>Check yourself</summary>

```
  mean                                           -> MAE 672.2  RMSE 681.4  MASE 11.96
  seasonal naive (last week)                     -> MAE  49.0  RMSE  58.5  MASE 0.87
  seasonal mean (last 4 weeks)                   -> MAE  57.8  RMSE  69.8  MASE 1.03
  decomposition (flat trend + index)             -> MAE  44.3  RMSE  56.0  MASE 0.79
```

The **mean** collapses: it's anchored to the network's average size
over two years, far below where it is now.
The **seasonal mean** now loses to the seasonal naive, because four
weeks ago is much further behind. Every baseline that holds the level
flat falls behind as the month goes on; growth is exactly what the
next lessons add.

</details>

## 6. Optional puzzle: seasonal naive with drift

Add a forecaster that copies last week and adds a weekly growth term:
week *k* ahead gets `k * g` extra rides, where `g` is an estimate of
growth per week. Try two estimates of `g`:

1. last week's mean minus the week before's mean;
2. last week's mean minus the mean of the same week a year earlier,
   divided by 52.

Add the forecasts just before the `for name, f in forecasts.items():`
loop and score them. Which is better, and why? The true growth is
0.55 rides a day, 3.85 a week.

<details>
<summary>Check yourself</summary>

```python
for label, g in [
    ("snaive + drift (1 week)", train[-7:].mean() - train[-14:-7].mean()),
    ("snaive + drift (52 weeks)", (train[-7:].mean() - train[-371:-364].mean()) / 52),
]:
    forecasts[label] = train[-7:][(h - 1) % 7] + g * ((h - 1) // 7 + 1)
```

The one-week estimate is `g = +26.1` rides per week, nearly seven times
the truth, and its MAE is **67.0**, much worse than the plain seasonal
naive. The year-on-year estimate is `g = +3.4`, close to the true 3.85,
and its MAE is **35.8**, a small win over the seasonal naive (37.5).
A growth rate estimated from one noisy week is mostly noise; spread
the same idea over a year and it starts to help. Estimating the trend
well from a long history is exactly what lesson 013's smoothing methods
do.

</details>
