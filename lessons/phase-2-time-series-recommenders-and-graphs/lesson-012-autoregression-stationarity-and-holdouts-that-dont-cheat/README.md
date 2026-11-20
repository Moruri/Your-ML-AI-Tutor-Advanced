# Lesson 012 - Autoregression, stationarity and holdouts that don't cheat

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 1 | Friday 2026-11-20

> **Goal:** Fit an AR model, test for stationarity, and evaluate with a
> rolling origin instead of a random split.

Time: about 60 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 011 set the bar: copy last week. Today we build the first model
that uses the past on purpose, an **autoregression** (AR): predict
today from a weighted sum of the last few days. It's the simplest idea
in forecasting, and it's still inside many of the models you'll meet
later.

Two questions decide whether an AR forecast deserves any trust, and
they're the real subject of this lesson:

1. **Is the series stable enough to learn from?** A model fitted to the
   past assumes the future behaves the same way. That's what
   *stationarity* means, and there's a test for it.
2. **Was it tested the way it will be used?** In real life you forecast
   the future from the past. A random train/test split lets the model
   peek at the future, and it can make a bad model look good.

Same rides as lesson 011: same function, same seed, same 700 days.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-012-autoregression-stationarity-and-holdouts-that-dont-cheat/lesson.py
```

It runs in about a second and saves `output/rolling_origin.png`: the
series we model on the left, error by days-ahead on the right.

## 1. What "stationary" means

A series is **stationary** when its behaviour doesn't depend on when you
look: the same mean, the same spread, the same relationship between
today and yesterday, in any window. Three toy series, 500 steps each,
split in half:

```
  white noise: mean, sd (1st | 2nd half)         -> -0.00, 0.94 | +0.03, 1.01
  AR(1), phi = 0.7: mean, sd (1st | 2nd half)    -> +0.01, 1.36 | +0.08, 1.36
  random walk: mean, sd (1st | 2nd half)         -> +2.27, 2.58 | +9.87, 3.97
  2000 walks: spread at step 100 / 400           -> 10.1 / 20.4
```

- **White noise:** independent draws. Stationary, and nothing to learn.
- **AR(1)** with phi = 0.7: each step is 0.7 times the last one plus
  fresh noise. It has memory, but it's always pulled back towards its
  mean, so both halves look alike. Stationary, and learnable.
- **Random walk** (phi = 1): each step is the last one plus noise. No
  pull back at all. The two halves have different means, and across
  2,000 walks the spread keeps growing (10 at step 100, 20 at step 400:
  it grows like the square root of time). Nothing about its level
  stays put, so the past can't tell you where it will settle.

The difference between 0.7 and 1 looks small and changes everything.
That's why we test for it.

## 2. A unit-root test you can read

The **augmented Dickey-Fuller** (ADF) test fits

```
change today = c + gamma * (yesterday's level) + a few lagged changes + noise
```

If gamma = 0, the level has no pull back: a random walk, also called
a **unit root**. If gamma < 0, a high level tends to be followed by a
fall: mean-reverting, stationary. So the test is just the t-statistic
on gamma. `adf_t()` in `lesson.py` is a dozen lines of least squares.

There's one catch, and it's the reason this test has its own name.
**Under the null, the t-statistic isn't normal.** We simulate 2,000
random walks to find the real 5% critical value:

```
  5% critical value: simulated / normal table    -> -2.88 / -1.64
  random walks 'rejected' using -1.64            -> 46.3%  (should be 5%)
```

Use the usual one-sided normal cutoff and you'd call almost half of all
random walks stationary. The simulated -2.88 matches the published
Dickey-Fuller table (-2.86 with a constant, for long series). Now the
tests:

```
  ADF t: AR(1), phi = 0.7                        ->  -9.48  stationary
  ADF t: random walk                             ->  -2.32  unit root not ruled out
  ADF t: daily rides, raw                        ->  -1.63  unit root not ruled out
  ADF t: rides, weekly difference                -> -11.67  stationary
  ADF t: rides minus trend/weekday/holiday       ->  -4.71  stationary
```

Raw rides fail: they trend upwards. There are two standard ways to fix
that, and both pass:

- **(a) Weekly difference:** model `rides today - rides same day last
  week`. Differencing removes both the trend and the weekly pattern.
- **(b) Remove what you know in advance:** regress rides on a straight-
  line trend, weekday dummies and a public-holiday flag (holidays are
  on the calendar, so this is fair), then model the residual. We call
  this the **calendar regression**. For the test, it's fitted on the
  first 672 days only.

Read "not ruled out" carefully. Failing to reject a unit root isn't
proof of one; with short or noisy series the test has little power.
Use it as one piece of evidence, together with a plot.

## 3. An autoregression on what's left

Fit AR(p) to the residual from (b) by least squares, choosing *p* by
**AIC** (fit error plus a penalty of 2 per coefficient, on the same rows
for every *p* so the comparison is fair):

```
  AIC picks p (residual, up to 14)               -> 6
  AR coefficients, lag 1..p                      -> +0.64 +0.02 +0.07 -0.04 +0.07 +0.08
  sd: residual / one-step AR error               -> 53.8 / 35.9
```

Almost all the weight sits on yesterday (+0.64; the simulation's
noise used 0.6). The small coefficients on later lags pick up the slow
rainy-season swings that a straight line and weekday dummies can't see.
The one-step error, 35.9, is close to the 35 we put in, so the AR has
found most of what is findable.

Approach (a) is less happy:

```
  AIC picks p (weekly difference, up to 21)      -> 21  (the maximum allowed)
```

When the trend is a steady line and the weekly pattern is fixed,
weekly differencing is **over-differencing**: it subtracts two noisy
days, which creates a pattern the AR then needs many lags to undo.
Difference when there's a real unit root; regress out a trend that's
deterministic. The ADF result on (b) is what tells you (b) is enough.

To forecast many days ahead, the AR predicts tomorrow, feeds that
prediction back in as "yesterday", and goes again (**recursive
forecasting**). The residual forecast fades towards zero, so far ahead
the model falls back to the calendar regression.

On lesson 011's holdout (the last 28 days):

```
  011 holdout MAE: seasonal naive                -> 37.5
  011 holdout MAE: weekly difference + AR(21)    -> 40.5
  011 holdout MAE: calendar regression + AR(6)   -> 36.9
```

The calendar regression + AR barely beats copying last week. Is it
actually better? One 28-day window can't tell us; lesson 011's exercise
4 showed the ranking change when the window moved. We need a better
test. First, a worse one.

## 4. A holdout that cheats

Lesson 011's degree-8 curve forecast the holdout with an MAE of 192.
Suppose you'd evaluated it the way you evaluate most ML models, with
random 5-fold cross-validation over all 700 days:

```
  degree-8 curve: random 5-fold CV MAE           -> 46.3
```

46 looks respectable, close to the seasonal naive. It's an illusion.
With a random split, every test day sits between training days on both
sides, so the curve only has to **interpolate**: fill a gap between
days it already knows. Forecasting means **extrapolating** past the
last day you have. Random CV measured a skill that the forecast never
uses.

The same leak shows up in quieter forms. Any feature computed with
future data (lesson 011's centred moving average, a scaler fitted on
the whole series, a "7-day average" that includes tomorrow) makes
random CV and even a single holdout look better than reality.

## 5. Rolling-origin evaluation

The honest test copies real use. Pick a forecast **origin**, fit only
on data before it, forecast the next 14 days, and score them. Then move
the origin forward and repeat. Here: 13 origins, one every two weeks,
from 2 June to 17 November 2025, with every model refitted at every
origin.

```
  first origin / last origin                     -> 2025-06-02 / 2025-11-17
  seasonal naive                                 -> MAE   54.6 | day 1 / 7 / 14:  60.2 /  80.8 /  75.4
  degree-8 curve                                 -> MAE  115.3 | day 1 / 7 / 14:  63.9 / 103.9 / 194.1
  weekly difference + AR(21)                     -> MAE   50.5 | day 1 / 7 / 14:  56.0 /  62.3 /  52.7
  calendar regression only                       -> MAE   42.0 | day 1 / 7 / 14:  46.8 /  47.9 /  50.0
  calendar regression + AR(6)                    -> MAE   40.2 | day 1 / 7 / 14:  30.3 /  47.5 /  45.7
  origins where it beats seasonal naive          -> 10 of 13
```

What the rolling origin tells us that one holdout couldn't:

- **The degree-8 curve is exposed.** 115 overall, 194 by day 14,
  against 46 under random CV. Its error grows with the horizon, the
  signature of a model that extrapolates badly.
- **The calendar regression + AR(6) really is better** than copying
  last week: 40.2 against 54.6, winning at 10 of 13 origins. One holdout
  had them almost tied.
- **The AR earns its keep at short horizons.** Day 1: 30.3 with the AR,
  46.8 without it. By day 14 the AR's memory has faded and the two are
  close. That's what an AR should do: yesterday matters for tomorrow,
  not much for two weeks from now.
- **The seasonal naive has a bad day 7 and day 14.** Those are Sundays,
  and at the first origin, "last Sunday" was Madaraka Day (1 June), so
  copying it predicted a holiday-sized dip two weeks running. Copying
  the past copies its accidents, too.

The curves on the right of the figure are bumpy because 13 origins
isn't many. With more data you'd use more origins (exercise 5), and you
would report the spread across origins, not just the mean.

Two refinements you'll see in practice:

- **Expanding vs sliding window.** Here each fit uses all history
  before the origin (expanding). A sliding window keeps only the last
  N days, which helps when the old data no longer describes the world.
- **A gap.** If your features use a 7-day trailing window, or if real
  data arrives a day late, leave a gap between training data and the
  first forecast day so the test matches what you'll actually know.

One honest confession: we chose *p* = 6 and *p* = 21 once, using data
up to November, which is after most of the origins. That's a small leak
of its own. Exercise 4 fixes it.

## What to remember

- **Stationary** means the series behaves the same in any window. AR
  models need it; trends and random walks break it.
- The **ADF test** asks whether there's a pull back to a mean. Its
  t-statistic needs **Dickey-Fuller critical values** (about -2.86 at
  5%), not -1.64.
- Make a series stationary by **regressing out** deterministic trend and
  calendar effects, or by **differencing** when there's a real unit
  root. Don't over-difference.
- An **AR model** is least squares on lagged values. Choose the order
  with AIC, forecast recursively, and expect its edge to be at short
  horizons.
- **Never evaluate a forecaster with a random split.** It measures
  interpolation. Use a **rolling origin**: refit at each origin, using
  only the past, and score many windows.
- Choose models, features and hyperparameters **inside** each origin,
  or you leak a little future into the test.

## What to do now

1. Run `lesson.py` and look at `output/rolling_origin.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 013 brings in the classic forecasters, exponential smoothing
   and Holt-Winters, and races them against the seasonal naive using the
   rolling origin you just built.
