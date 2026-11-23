# Lesson 013 - Exponential smoothing and classic forecasters

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 2 | Monday 2026-11-23

> **Goal:** Forecast with Holt-Winters and compare it fairly against a
> naive seasonal baseline.

Time: about 60 minutes to read and run, plus the exercises. Nothing new
to install.

---

Before machine learning arrived in demand planning, there was
**exponential smoothing**, and in a lot of warehouses there still is.
It was invented in the 1950s to forecast spare parts for the US Navy,
it needs a few lines of arithmetic per day, and it's annoyingly hard to
beat. Any forecasting tool you buy will have a version of it inside.

Today we build the family from the ground up:

1. **Simple exponential smoothing (SES):** a running average that
   forgets old days at a steady rate.
2. **Holt's method:** add a trend, and learn to distrust it (the
   **damped** trend).
3. **Holt-Winters:** add a weekly season. That's the one the ops team
   will actually use.

Then we race them against the seasonal naive and lesson 012's
calendar regression + AR, on the same 13 rolling origins. Same rides as
lessons 011 and 012: same function, same seed, same 700 days.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-013-exponential-smoothing-and-classic-forecasters/lesson.py
```

It runs in about five seconds and saves `output/holt_winters.png`: the
last forecast with its 80% interval on the left, error by days-ahead
for every model on the right.

## 1. An average that forgets

SES keeps one number, the **level**, and nudges it towards each new
day:

```
level today = alpha * (today's value) + (1 - alpha) * (level yesterday)
forecast for any future day = level today
```

Unroll that and the level is a weighted average of the past, with
weights alpha, alpha(1 - alpha), alpha(1 - alpha)^2, ... shrinking
geometrically. That's the "exponential" in the name.

```
  alpha = 0.1: weights on last 5 points          -> 0.100 0.090 0.081 0.073 0.066  (half-life 6.6)
  alpha = 0.5: weights on last 5 points          -> 0.500 0.250 0.125 0.062 0.031  (half-life 1.0)
```

With alpha = 0.1 the last day gets 10% of the weight and the memory
has a half-life of about a week. With alpha = 0.5, half the weight sits
on the last day. Which is better? It's a trade-off, and a toy series
shows it: a level of 100 that jumps to 130 halfway through, plus noise
with sd 10.

```
  alpha = 0.1: steady MAE / days to catch up     -> 8.0 / 10
  alpha = 0.5: steady MAE / days to catch up     -> 8.8 / 3
  alpha fitted by one-step squared error         -> 0.16
```

A small alpha averages away more noise when nothing is changing; a big
alpha catches up with a real change faster. You don't pick alpha by
taste: you pick the value that makes the **one-step-ahead errors**
smallest on the history you have. That's how every smoothing weight in
this lesson is fitted, using `scipy.optimize.minimize`, and the data
only ever comes from before the forecast origin.

## 2. Holt: a trend, and a reason to doubt it

Holt's method adds a second state, the **slope**, smoothed the same way
with its own weight beta. The forecast h days ahead is
`level + h * slope`: a straight line that never stops.

That's brave. A slope fitted on the last few months, extended for a
year, is how forecasts get famous for the wrong reasons. So Gardner and
McKenzie added a **damping** factor phi (just under 1): the forecast
adds `phi + phi^2 + ... + phi^h` slopes, which levels off at
`phi / (1 - phi)` slopes however far you go.

Fitted on the first 672 days (with the weekly season from section 3
included; more on that in a moment):

```
  HW: alpha, beta, gamma, phi                    -> 0.27, 0.007, 0.28, 1.000
  HW, damped: alpha, beta, gamma, phi            -> 0.25, 0.000, 0.27, 0.870
  HW: slope / gain over 1 year                   -> +0.47 / +168 rides
  HW, damped: slope / gain over 1 year           -> +0.00 / +0 rides
  truth: slope / gain over 1 year                -> +0.55 / +197 rides
```

Two things to read here.

- **beta is tiny** (0.007 and 0). The rides grow by a steady 0.55 a
  day, and a steady slope is best learned slowly. A big beta would chase
  every noisy week.
- **The damped version gave up on the trend.** Its optimiser found that
  one-step errors are just as small with no slope at all: over one day,
  0.55 rides of growth is invisible next to noise of 35. So it set
  beta = 0, and its year-ahead forecast is flat. The undamped version
  kept a slope and gets the year roughly right (+168 vs +197).

That's an important limit of fitting by one-step error: it tells you
what helps **tomorrow**, and says little about next year. For a 14-day
plan it hardly matters, as section 4 shows. For a budget, a trend you
can defend with outside knowledge beats whatever the optimiser chose.

## 3. Holt-Winters: add the week

**Holt-Winters** adds a third state: seven **seasonal** numbers, one
per weekday, each smoothed with weight gamma. Every day updates the
level, the slope, and that weekday's seasonal number. The forecast is

```
forecast h days ahead = level + (damped) slope + seasonal number for that weekday
```

The code is `hw_filter()` and `hw_forecast()` in `lesson.py`: about 25
lines with no library. Starting values come from the first two weeks.

```
  learned weekly pattern, Mon..Sun               -> +24 +93 +103 +79 +55 -114 -239
  true weekly pattern, Mon..Sun                  -> +60 +75 +80 +75 +70 -110 -250
  one-step error sd: SES / HW / HW damped        -> 136.0 / 62.4 / 62.1
```

The learned week is close to the truth, except **Monday**, which is
about 36 rides short. The culprit is Mashujaa Day, Monday 20 October
2025, three weeks before the end of training. Holt-Winters has no idea
it was a holiday. It saw a Monday 300 rides down and moved its Monday
number down by a gamma-sized share of that, and the memory hasn't
faded yet. Lesson 012's calendar regression had a holiday flag; Holt-
Winters, as written, cannot. Exercise 4 fixes it.

SES, with no season, has one-step errors twice as big: it forecasts the
same number for a Tuesday and a Sunday.

On lesson 011's holdout (the last 28 days):

```
  011 holdout MAE: SES                           -> 90.0
  011 holdout MAE: Holt-Winters                  -> 36.8
  011 holdout MAE: Holt-Winters, damped          -> 38.3
  011 holdout MAE: seasonal naive (lesson 011)   -> 37.5
```

A tie with copying last week. As lesson 012 taught us, one window
can't settle that.

## 4. A fair race

Lesson 012's rolling origin: 13 origins, one every two weeks from 2
June to 17 November 2025, 14 days ahead each, and **every model's
weights refitted at every origin** on that origin's past only. We also
bring back lesson 012's winner, and add one more contestant: the plain
average of Holt-Winters damped and the calendar regression + AR.

```
  seasonal naive                                 -> MAE  54.6 | day 1 / 7 / 14:  60.2 /  80.8 /  75.4
  SES                                            -> MAE 112.2 | day 1 / 7 / 14: 117.1 / 211.3 / 187.0
  Holt-Winters                                   -> MAE  49.0 | day 1 / 7 / 14:  47.6 /  61.7 /  54.5
  Holt-Winters, damped                           -> MAE  48.6 | day 1 / 7 / 14:  47.9 /  61.2 /  57.3
  calendar regression + AR(6)                    -> MAE  40.2 | day 1 / 7 / 14:  30.3 /  47.5 /  45.7
  average of HW damped and reg + AR              -> MAE  41.7 | day 1 / 7 / 14:  26.2 /  54.4 /  48.9
  beats seasonal naive: HW, damped               -> 8 of 13 origins
  beats seasonal naive: calendar reg. + AR(6)    -> 10 of 13 origins
```

What the race says:

- **Holt-Winters beats copying last week**, 48.6 against 54.6, and
  it's much steadier at day 7 and day 14 because it averages many weeks
  instead of trusting one (remember the Madaraka Day echo). But it wins
  at only 8 of 13 origins: a real gain, not a landslide.
- **Damped or not hardly matters at 14 days** (48.6 vs 49.0). Over two
  weeks the slope adds about 7 rides. Over a year it's the whole story.
- **SES is the wrong tool here.** Without a season, it forecasts a flat
  line, and day 7 and day 14 (Sundays) are 200 rides off. Use SES for
  series with no strong cycle, such as weekly totals.
- **Lesson 012's model still wins**, 40.2. It knows two things Holt-
  Winters doesn't: the holidays, and a straight trend fitted to all of
  history. Holt-Winters adapts to anything, which is its strength when
  the world shifts and its weakness when the world is this regular.
- **The average is best on day 1** (26.2, against 30.3 and 47.9).
  Averaging forecasts from different kinds of model is one of the most
  reliable tricks in forecasting: their errors partly cancel. Overall
  it doesn't beat the better model here (41.7 vs 40.2), but it's never
  far behind either, which is a good property when you don't know which
  model will be better next month.

## 5. How wide is the fog?

A planner needs a range, not just a number. Two ways to put an 80%
interval around Holt-Winters damped:

- **Flat:** forecast plus or minus 1.28 times the sd of the one-step
  errors, the same width every day.
- **Simulated paths:** run the update rules forward 14 days, 500 times,
  each day adding an error drawn at random from the model's own past
  one-step errors (a **bootstrap**). Take the 10th and 90th percentile
  of the paths each day. Errors feed into the level, so the paths fan
  out with the horizon, as they should.

The check: across the 13 origins, how many real days fell inside?

```
  flat +/- 1.28 sd                               -> all 84% | days 1-3 90% | days 12-14 79%
  simulated paths                                -> all 84% | days 1-3 85% | days 12-14 87%
  last origin: width day 1 / day 14 (simulated)  -> 141 / 204 rides
```

Both land near 80% overall, a little wide, partly because the one-step
errors include holidays, which make ordinary days look noisier than
they are. The difference is in the shape. The flat interval is too wide
early (90%) and its coverage slides as the horizon grows; the simulated
one holds steady, because it widens where it should: 141 rides wide on
day 1, 204 on day 14 at the last origin. Uncertainty grows with the
horizon, and simulating the model is the honest way to show it. With 13 origins these percentages are rough (exercise 5 in
lesson 012 explains how to get more), but the direction is the lesson.

## What to remember

- **Exponential smoothing** is a weighted average with geometrically
  fading weights. alpha trades noise reduction against speed of
  reaction; fit it by one-step-ahead error on the past.
- **Holt** adds a slope, **Holt-Winters** adds a season. Each is a
  few update rules, and the whole state fits in a dozen numbers.
- A **damped trend** stops a forecast from running off to infinity.
  Fitting by one-step error can't judge long-run trends, so bring
  outside knowledge for long horizons.
- Smoothing models **don't know the calendar**: a holiday leaks into
  the seasonal numbers. Tell them, or clean the history first.
- Race forecasters on a **rolling origin**, refitting every time.
  Here Holt-Winters beats the seasonal naive (48.6 vs 54.6), and the
  calendar regression + AR still beats both (40.2).
- **Average different models** when you're unsure; it rarely hurts.
- Give a range, and **check the range**. Intervals should widen with
  the horizon.

## What to do now

1. Run `lesson.py` and look at `output/holt_winters.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 014 turns the forecast into a tabular problem: calendar,
   lag and weather features in a gradient-boosted model, and an honest
   look at when that beats everything in this lesson.
