# Lesson 011 - Time series as a modelling problem

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 1 | Friday 2026-11-20

> **Goal:** Decompose a series into trend, seasonality and remainder, and
> set a forecasting baseline that is hard to beat.

Time: about 45 minutes to read and run, plus the exercises. Nothing new
to install.

---

Welcome to phase 2. In phase 1 every row was its own little world: one
rider, one station, one coin flip. Shuffle the rows and nothing changed.
From today, the rows come **in order**, and the order is half the
information. This phase covers three kinds of data like that: series in
time (lessons 011-014), preferences between people and things
(015-017), and networks (018-019).

We start with the question every operations team asks on a Friday:
**how many rides will we have over the next four weeks?** Bikes need
moving, batteries need charging, and the mechanics want to plan their
leave. Before we build anything clever, we'll take the series apart to
see what it's made of, and then set up a few forecasts so simple they
feel like cheating. They are the bar. Most of the time, the clever model
you build next has to fight hard to clear it.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-011-time-series-as-a-modelling-problem/lesson.py
```

It runs in about a second and saves `output/decomposition.png` next to
the script: the series and its trend, the weekly pattern, what's left
over, and the forecasts against what really happened.

## 1. Order is information

The data is simulated, so we know the truth: 700 days of network-wide
rides from Monday 2024-01-01 to Sunday 2025-11-30. Inside it: steady
growth (about 0.55 extra rides a day, every day), a strong weekly rhythm
(commuters on weekdays, quiet Sundays), dips in the long rains
(April-May) and the short rains (November), public holidays, and noise
that's **sticky**: a busy day tends to be followed by a busy day.

```
  days / first / last                            -> 700 / 2024-01-01 / 2025-11-30
  mean rides, first 8 weeks / last 8 weeks       -> 905 / 1224
  correlation with yesterday: real / shuffled    -> 0.58 / -0.00
  correlation with a week ago: real / shuffled   -> 0.88 / -0.02
```

Today's rides look a lot like last Tuesday's rides when today is a
Tuesday (0.88), and quite a lot like yesterday's (0.58). Shuffle the
days and both correlations vanish, even though every single number is
still there. That's what "the order is information" means, and it has
two consequences that run through this whole phase:

- **Past values are features.** The best predictor of next Tuesday is
  often last Tuesday.
- **Rows are not independent.** Anything that assumes they are (a
  random train/test split, a plain standard error) will quietly lie to
  you. Lesson 009 met the standard error version; lesson 012 meets the
  train/test version.

## 2. Take it apart: trend, season, remainder

The classical **additive decomposition** writes each day as

```
rides = trend-cycle + weekly pattern + remainder
```

and finds each piece with nothing more than averages:

1. **Trend-cycle:** a centred 7-day moving average. Averaging exactly
   one week wipes out the weekly pattern and leaves the slow movement
   (growth and the rainy seasons, which is why it's a trend-*cycle*).
2. **Weekly pattern:** subtract the trend, then average what's left for
   each weekday. Centre the seven numbers so they sum to zero.
3. **Remainder:** whatever is left.

```
  weekly index, Mon..Sun                         -> +64 +80 +75 +65 +71 -108 -247
  share of variance: trend / weekly / remainder  -> 46% / 46% / 7%
  trend-cycle values missing at each end         -> 3 days
  remainder: correlation with yesterday          -> 0.03
  three most negative remainders fall on         -> 2024-12-12, 2024-12-26, 2025-01-01
```

The weekly index says a typical Tuesday is 80 rides above the trend
and a typical Sunday 247 below (the simulation used +75 and -250, so
the averages recover it well). Trend and weekly pattern each explain
about half the variation; the remainder is only 7%. The most negative
remainders are Jamhuri Day, Boxing Day and New Year's Day: the
decomposition doesn't know about holidays, so it leaves them for us to
find. That's a feature, not a bug. **The remainder is where surprises
live**, and it's the first place to look for outages, data errors and
events. (Look at the bottom-left of the figure: each holiday's big dip
comes with small *positive* bumps on the days around it, because the
holiday drags the moving average down for a week.)

Two details matter later:

- **The ends are missing.** A centred average at day *t* needs three
  days after *t*. So at the last day of your data, exactly where a
  forecast starts, the trend is unknown. Any feature built with a
  centred window quietly uses the future. Remember this when someone
  shows you a forecast that is suspiciously good.
- **The remainder barely remembers yesterday (0.03)**, even though the
  noise we simulated is sticky. The 7-day moving average follows slow
  wiggles so closely that it absorbs most of that stickiness into the
  "trend". Decomposition is a lens for *looking*; it's not a forecast
  model. Lesson 012 models the memory properly.

Additive suits series whose weekly swing stays roughly the same size as
the level grows. If the Sunday dip grew in proportion to the business,
you'd use a **multiplicative** decomposition (divide instead of
subtract), or decompose the log of the series.

## 3. Baselines: the forecasts to beat

Hold out the last 28 days (3-30 November 2025), forecast them from the
672 days before, and compare:

- **mean:** every future day is the average of the past.
- **naive:** every future day equals the last day you saw.
- **drift:** the naive forecast plus the average daily change over the
  whole history, a straight line from the first day to the last.
- **seasonal naive:** each future day equals the same weekday last
  week.
- **seasonal mean:** each future day equals the average of the same
  weekday over the last 4 weeks.
- **decomposition:** the last trend-cycle value we can compute, held
  flat, plus the weekly index.

```
  mean                                           -> MAE 180.7  RMSE 198.3  MASE 3.26
  naive (last value)                             -> MAE 239.5  RMSE 262.1  MASE 4.33
  drift                                          -> MAE 232.8  RMSE 255.7  MASE 4.21
  seasonal naive (last week)                     -> MAE  37.5  RMSE  46.2  MASE 0.68
  seasonal mean (last 4 weeks)                   -> MAE  46.6  RMSE  53.2  MASE 0.84
  decomposition (flat trend + index)             -> MAE  34.5  RMSE  41.8  MASE 0.62
  MASE scale: in-sample seasonal-naive MAE       -> 55.4
```

Three of these ignore the week, and all three are terrible. The naive
forecast is the worst because the last training day was a Sunday, so it
predicts a Sunday every day for a month. **Any baseline for a seasonal
series has to be seasonal.** The seasonal naive is off by 37.5 rides a
day, about 3% of a typical day; the decomposition forecast edges it at
34.5. The seasonal mean is worse than last week alone, because four
weeks ago the network was smaller and the short rains hadn't started.
Averaging more history only helps if the history is still relevant.

About the three columns:

- **MAE** (mean absolute error) is in rides. Easy to explain to the
  operations team.
- **RMSE** punishes big misses more. Use it when one terrible day costs
  more than several slightly-off ones (running out of bikes at a
  stadium, say).
- **MASE** divides the MAE by the MAE that a seasonal naive forecast
  gets *inside the training data* (55.4 here). Below 1 means "better
  than copying last week, on average". It's unit-free, so you can
  compare it across stations of very different sizes, which plain MAE
  can't do, and it doesn't blow up near zero the way percentage errors
  do.

## 4. A clever model that loses

Now try something that looks more serious: a least-squares fit of a
polynomial in time plus a dummy for each weekday, using all 672 days.

```
  poly degree 1 + weekday: in-sample / test MAE  -> 48.8 / 39.9
  poly degree 8 + weekday: in-sample / test MAE  -> 44.8 / 192.3
```

The degree-8 curve fits the past *better* (44.8 against 48.8) and
forecasts the future five times *worse*. A high-degree polynomial bends
to follow the rainy seasons in the history, and past the last data
point nothing holds it down: it plunges (the green line, bottom-right of
the figure). The straight line is sensible but still loses to copying
last week, because it ignores the short rains and the sticky noise.

That's the lesson in one picture. **In-sample fit tells you almost
nothing about forecasting**, flexible functions of time extrapolate
badly, and a baseline you can write in one line is a high bar.

## What to remember

- In a time series, **order carries information**. Past values are
  features, and rows aren't independent.
- **Decompose before you model.** Trend-cycle, seasonal pattern and
  remainder tell you what any model has to capture, and the remainder
  shows you the surprises.
- **Centred windows use the future.** Fine for looking, never for
  forecasting features.
- **Always set baselines first**, and make them seasonal when the data
  is. The seasonal naive is the one to beat.
- Report **MASE** next to MAE so "good" has a reference point.
- A better in-sample fit can mean a worse forecast.

## What to do now

1. Run `lesson.py` and look at all four panels of
   `output/decomposition.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 012 builds the first real forecaster, an autoregression. It
   asks whether the series is stable enough to model, and replaces
   today's single holdout with a test that can't cheat.
