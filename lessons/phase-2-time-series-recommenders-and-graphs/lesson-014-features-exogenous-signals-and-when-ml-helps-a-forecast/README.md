# Lesson 014 - Features, exogenous signals and when ML helps a forecast

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 2 | Monday 2026-11-23

> **Goal:** Add calendar and lag features to a tabular model and see when
> it beats the classical methods.

Time: about 60 minutes to read and run, plus the exercises. Nothing new
to install (`scikit-learn` is already in the shared stack).

---

Every forecaster so far has looked only at the rides. But the ops team
knows things the series doesn't: it's going to rain on Saturday, there's
a match at the stadium, Monday is a public holiday. Signals from outside
the series are called **exogenous**, and they're where most real
forecasting gains come from.

Once you have them, the forecast stops looking like a special
time-series problem and turns into an ordinary table: one row per
future day, some columns, a target. Hand that table to gradient-boosted
trees and you have the method that wins a lot of forecasting
competitions. It's also the method that most often lies about how good
it is, because building the table honestly is fiddly. So today is half
modelling and half bookkeeping.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-014-features-exogenous-signals-and-when-ml-helps-a-forecast/lesson.py
```

It runs in about fifteen seconds and saves `output/features.png`: error
on dry and wet days for each model on the left, the last two weeks of
forecasts with the wet days shaded on the right.

## 1. A rainier city

Lessons 011-013 hid the weather inside a smooth seasonal dip, which is
kind: smooth things are easy to forecast. Real rain falls on some days
and not others. So today's city is the same as before (same growth, same
week, same holidays, same sticky noise) except for two things:

- **Daily rainfall.** Wet days are more likely in the long and short
  rains. Rain cuts rides, but the effect **saturates** (a downpour isn't
  ten times worse than a shower), and it's **1.7 times bigger at
  weekends**, when people ride for fun rather than to get to work.
- **Match days** at the stadium, mostly at weekends: about +180 rides,
  but only if it's dry. Nobody cycles to a wet match.

```
  wet days (>= 2 mm) / match days                -> 17% / 55
  weekday: mean rides dry / wet                  -> 1170 / 1046  (gap 124)
  weekend: mean rides dry / wet                  -> 914 / 679  (gap 236)
```

Those are **interactions**: the effect of rain depends on the weekday,
and the effect of a match depends on the rain. Remember that; it
decides who wins in section 3.

And here's the catch. You don't know next Saturday's rain. You know the
**weather forecast** for it, and that gets worse the further ahead it
looks:

```
  rain forecast, 1 day(s) ahead: wet days caught -> 92%, false alarms 12%
  rain forecast, 3 day(s) ahead: wet days caught -> 87%, false alarms 17%
  rain forecast, 7 day(s) ahead: wet days caught -> 77%, false alarms 31%
```

The simulation keeps the full archive: for every day, the forecast that
was issued 1, 2, ... 7 days before. If you only stored the latest
forecast for each day, you couldn't build honest training data. That's
a real-world lesson too: **save your forecasts of the inputs**, not just
the inputs.

## 2. From a series to a table

We forecast 7 days ahead, and build one row per **(target day, lead
time)**: "standing on day d - h, forecast day d". The rule for every
column is simple to say and easy to break: **it must be known on day
d - h.**

| Column | Known in advance? |
|---|---|
| trend, weekday, holiday, match, day-of-year (as sine and cosine) | Yes: it's the calendar and the fixture list |
| `lag7`: rides 7 days before the target | Yes, for any lead up to 7 |
| `lag7_13_mean`: average of rides 7 to 13 days before | Yes, for any lead up to 7 |
| `rain_fc`: the rain forecast issued h days before | Yes, that's what a forecast is |
| actual rain on the target day | **No** |
| rides yesterday (1 day before the target) | **Only when h = 1** |

Every lead gets its own row, with a `lead` column, so one model learns
all seven horizons and knows that a 7-day rain forecast is less
trustworthy than a 1-day one. This is called the **direct** strategy
(one model forecasting each horizon straight away), as opposed to
lesson 012's **recursive** AR, which fed its own forecasts back in.

```
  training rows at the first origin, 2025-06-02  -> 3528 rows = 504 days x 7 leads
  one row: Sat 2025-11-29 at lead 3              -> made on Wed 2025-11-26, knowing:
      trend              1.00
      dow                5.00
      holiday            0.00
      match              0.00
      doy_sin           -0.54
      doy_cos            0.84
      lead               3.00
      lag7            1219.00
      lag7_13_mean    1334.29
      rain_fc            0.00
```

One more step before the trees. **Trees can't extrapolate.** A tree
predicts the average of training days that landed in the same leaf, so
it can never forecast more rides than it has seen. With a growing
series that's a built-in under-forecast. The fix is the same idea as
lesson 012's: take out a straight-line trend (fitted on the past only),
let the model learn what's left, and add the line back.

## 3. The race

Weekly origins over the last 26 weeks (every Monday from 2 June to 24
November 2025), 7 days ahead each, every model refitted at every origin.
The tree settings are deliberately plain: 100 shallow trees, at least
40 days per leaf, because a few hundred days isn't much data. Tuning
them on these same windows would be one more leak; if you tune, do it
inside each origin, as lesson 012's exercise 4 did for the AR order.

```
  seasonal naive                                 -> MAE  89.3 | day 1 / 7:  72.9 / 169.2 | bias  -3.9
  calendar regression + AR(6)                    -> MAE  61.5 | day 1 / 7:  46.4 /  95.2 | bias -10.7
  linear, all features                           -> MAE  45.6 | day 1 / 7:  36.8 /  49.7 | bias  -7.6
  boosted trees, raw rides                       -> MAE  53.7 | day 1 / 7:  44.1 /  63.0 | bias -26.4
  boosted trees, detrended                       -> MAE  45.5 | day 1 / 7:  35.2 /  51.9 | bias  -5.8
  boosted trees, no rain column                  -> MAE  57.5 | day 1 / 7:  41.8 /  82.9 | bias  -7.9
  boosted trees, actual rain (oracle)            -> MAE  37.9 | day 1 / 7:  35.9 /  42.9 | bias  -5.0
```

Read it in this order:

- **The features did the work.** Lesson 012's champion knows holidays
  but not rain or matches: 61.5. Give the models a rain forecast and the
  fixture list and both linear (45.6) and trees (45.5) cut a quarter of
  the error. Take the rain column away from the trees and they fall
  back to 57.5. The exogenous signal mattered far more than the choice
  of algorithm.
- **Linear and trees tie overall.** With 500-odd days of history, the
  trees' ability to find interactions roughly pays for their extra
  noise, and no more.
- **Raw trees under-forecast**, by 26 rides a day on average: the
  extrapolation problem, exactly as predicted. Detrending fixes it.
- **The seasonal naive is now terrible** (89.3, and 169 on day 7). When
  rain decides a weekend, copying last weekend copies last weekend's
  weather.

Where does the error come from?

```
  MAE dry / wet days: calendar regression + AR(6) -> 51.6 / 141.7
  MAE dry / wet days: linear, all features       -> 41.4 / 79.4
  MAE dry / wet days: boosted trees, detrended   -> 43.4 / 62.4
```

That's the tie, opened up. **On wet days the trees win clearly** (62.4
vs 79.4): they've learned that rain bites harder at weekends and that it
cancels a match, which a straight-line rain coefficient can't express.
**On dry days the linear model wins** (41.4 vs 43.4): when nothing
unusual happens, its smooth, additive structure is exactly right and
the trees' steps just add noise. Since most days are dry, it evens out.

## 4. Two ways to flatter a forecast

Two experiments that show how a backtest can look better than reality.

**Yesterday's rides.** Add one more column, the most recent rides, in
two ways: honestly (the last day before the origin, so the lag equals
the lead) and leakily (the day before the target, which at leads 2-7
hasn't happened yet when the forecast is made).

```
  trees + newest known day (lag = lead)          -> MAE  45.6 | day 1 / 7:  34.8 /  52.0
  trees + yesterday at every lead (LEAK)         -> MAE  43.5 | day 1 / 7:  36.2 /  58.0
```

The honest column barely helps; the leak "improves" the model by two
rides a day, and the improvement would vanish in production. Here the
leak is modest because the noise forgets within a couple of days. In
real data with strong day-to-day momentum (sales after a promotion,
traffic during a strike), the same mistake can halve the backtest error
of a model that's no better at all. It's the commonest bug in
ML forecasting. If a column is computed from the target series, check
its lag against the lead, row by row.

**Tomorrow's weather.** The oracle row in section 3 trained and tested
with the *actual* rain:

```
  trees with actual rain vs rain forecast        -> 37.9 vs 45.5
```

That's not a model you can run, but it's a useful number: it's the
**value of a perfect weather forecast**, 7.6 rides a day, and the
ceiling for anything you'd pay to improve it. The opposite use is the
dangerous one. Many published "weather improves our forecasts by X%"
results quietly used observed weather. Ask which one was used.

## 5. So when do the trees help?

Two more races. First, lessons 011-013's rides: smooth seasons, no daily
rain, no matches. Nothing nonlinear to find.

```
  calendar regression + AR(6)                    -> MAE  37.4 | day 1 / 7:  30.1 /  45.0
  linear, all features                           -> MAE  39.7 | day 1 / 7:  40.6 /  47.6
  boosted trees, detrended                       -> MAE  40.8 | day 1 / 7:  37.9 /  45.7
```

The classical model wins. It's correctly specified, and its AR uses
the newest day through recursion, which the table models only see at
lag 7. The trees bring nothing to a problem that's already linear.

Second, the rainy city again, but with more history. These are new
random draws of the same city, so compare models within a block, not
across blocks.

```
5b. The same rainy city, with 1400 days of history
  calendar regression + AR(6)                    -> MAE  71.1 | day 1 / 7:  62.1 / 102.1
  linear, all features                           -> MAE  55.5 | day 1 / 7:  46.9 /  79.0
  boosted trees, detrended                       -> MAE  51.2 | day 1 / 7:  35.1 /  64.9

5b. The same rainy city, with 2800 days of history
  calendar regression + AR(6)                    -> MAE  78.1 | day 1 / 7:  80.0 / 121.9
  linear, all features                           -> MAE  60.8 | day 1 / 7:  62.1 /  81.5
  boosted trees, detrended                       -> MAE  56.5 | day 1 / 7:  55.3 /  70.8
```

With four or eight years of history, the trees pull clearly ahead of
the linear model (by about 4 rides a day). There are now enough wet
weekends and wet match days for them to learn the interactions
properly.

Putting it together, ML earns its place in a forecast when:

1. There are **signals from outside the series** (weather, events,
   prices, promotions) that you can know or forecast in advance.
2. Those signals act **nonlinearly or in interactions**, so a
   straight-line model can't express them.
3. There's **enough data** to learn those shapes: long history, or many
   related series (every station at once, not just the network total).

Missing any of those, a well-built classical model is as good or
better, cheaper to run, and easier to explain. And even when ML wins,
look at where: here, on wet days. That's where the extra complexity is
paid for.

## What to remember

- **Exogenous signals** usually matter more than the algorithm. Here a
  rain forecast cut a quarter of the error.
- Build a forecast table with one row per **(target, lead)**, and check
  every column: is it known on day d - h? Lags must be at least the
  lead; weather must be the forecast, not the observation.
- **Store your input forecasts** so you can build honest training data.
- **Trees can't extrapolate.** Remove the trend first and add it back.
- Leaks make backtests look better than reality. **An oracle** run is
  fine as a ceiling, never as a result.
- Trees win when there are nonlinear effects, interactions and enough
  data. Otherwise, a linear model with good features is hard to beat.
- Look at errors **by segment** (wet/dry, weekday/weekend), not just
  the overall MAE. A tie overall can hide a clear winner on the days
  that matter.

## What to do now

1. Run `lesson.py` and look at `output/features.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. That closes the time-series thread. Lesson 015 starts on
   recommender systems: framing recommendation as predicting and
   ranking, and measuring it with ranking metrics instead of accuracy.
