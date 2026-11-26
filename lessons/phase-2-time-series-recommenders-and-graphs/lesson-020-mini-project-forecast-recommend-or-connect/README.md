# Lesson 020 - Mini-project: forecast, recommend or connect

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 5 | Thursday 2026-11-26

> **Goal:** Take one of the three threads to a short, evaluated
> write-up on a small public dataset.

Time: about 90 minutes to read and run, plus your own track and
write-up. Nothing new to install.

---

Phase 2 gave you three ways of looking at data that a plain table
hides: **order** (time series), **who likes what** (recommenders) and
**who's connected to whom** (graphs). Today you pick one and do the
whole job, start to finish, the way you would at work: a question
someone actually has, a plan written before you look at results, a
baseline you must beat, an evaluation that can't cheat, and a short
note that ends with what to do.

The three tracks:

- **A. Forecast** (worked in full below): next week's daily rentals
  for a real bike-share system, Capital Bikeshare in Washington DC,
  2011-2012.
- **B. Recommend** (starter): which stations will a rider use next?
- **C. Connect** (starter): Zachary's karate club, a classic public
  social network. Predict who sided with whom when the club split, and
  which friendships exist.

Read track A even if you plan to do B or C. The method is the same;
only the data changes.

## The data

Track A uses the UCI **Bike Sharing** dataset (daily file, 731 rows),
saved next to the script as `day.csv`. It's real: daily rental counts
from Capital Bikeshare with the calendar and the weather for each day.
It's published under CC BY 4.0; the citation is Fanaee-T, H. and
Gama, J. (2013), *Event labeling combining ensemble detectors and
background knowledge*, Progress in Artificial Intelligence,
doi:10.1007/s13748-013-0040-3.

The karate club for track C is small enough to sit in the script as
an edge list (Zachary, 1977: 34 members, 78 friendships). Track B uses
simulated riders so it runs anywhere; exercise 6 suggests a real
dataset if you want one.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-020-mini-project-forecast-recommend-or-connect/lesson.py
```

It takes about half a minute (it refits every model for every week of
the backtest) and saves `output/forecast.png`: the sealed test weeks
with forecasts and ranges on the left, error by horizon on the right.

As in lesson 010, read each block of output and **say what you'd
conclude before reading my take**.

## 0. Write the plan before you look

```
  question                                           -> total rentals on each of the next 7 days
  who uses it                                        -> ops lead: van shifts and maintenance days
  metric                                             -> MAE in rentals/day, plus 80% interval coverage
  forecast made                                      -> every Sunday night, for Mon-Sun
  model selection on (weekly origins)                -> 2012-03-05 to 2012-11-04  (35 weeks)
  sealed test, used once (weekly origins)            -> 2012-11-05 to 2012-12-30  (8 weeks)
  beat this or don't ship                            -> seasonal naive (same weekday last week)
```

Every line here is a decision that's easy to fudge later, so it's made
first:

- **The question comes from a user.** The ops lead plans van shifts a
  week ahead, so the forecast is made on Sunday for the next seven
  days. That fixes the horizon (1 to 7 days) and what we're allowed to
  know: nothing after Sunday night.
- **MAE in rentals per day** is a number the ops lead can picture.
  Intervals matter as much as the point forecast, because staffing is
  planned for a bad day, not an average one.
- **Two time periods, never shuffled** (lesson 012). Models are
  compared on 35 weekly **rolling origins** from March to October
  2012: at each Sunday, refit on everything up to that day, forecast
  the next week, move on. November and December 2012 are **sealed**
  and opened once, at the end, for the chosen model only.
- **The baseline is named in advance.** "Same as last week" is what
  the ops lead does today. If we can't beat it, the answer is "keep
  doing that".

## 1. Look at the data (training part only)

```
  days / first / last                                -> 731 / 2011-01-01 / 2012-12-31
  mean rentals per day: 2011 / 2012                  -> 3406 / 5600  (+64%)
  before 2012-03: mean by weekday (Sun..Sat)         -> 3295 3403 3515 3271 3442 3521 3358
  before 2012-03: mean on clear / misty / wet days   -> 3683 / 3072 / 1705
  before 2012-03: autocorrelation at lag 1 / lag 7   -> 0.78 / 0.67
```

Three things to take away, all from data before the first forecast:

- **Strong growth.** 2012 is 64% busier than 2011. Anything that
  predicts raw levels from a long-ago average will undershoot, so the
  models predict each day **relative to the last week's level**.
- **A weak weekly pattern in the total.** The weekday means barely
  differ. That's surprising for a bike-share system, and it hides
  something: commuters ride on weekdays and tourists at weekends, and
  in the total they nearly cancel. Exercise 4 splits them.
- **Weather matters a lot.** Wet days have under half the rentals of
  clear days. But we won't know next Saturday's weather on Sunday, so
  we must be careful how we use it.

## 2. Baselines and candidates

```
  naive (last day)                                   -> 1197   skill vs seasonal naive -10%
  seasonal naive                                     -> 1090   skill vs seasonal naive +0%
  ridge: calendar + lags                             -> 917   skill vs seasonal naive +16%
  ridge: + forecast weather                          -> 923   skill vs seasonal naive +15%
  boosting: + forecast weather                       -> 855   skill vs seasonal naive +22%
  ridge: + PERFECT weather (not allowed)             -> 846   skill vs seasonal naive +22%
  chosen (best honest model on selection weeks)      -> boosting: + forecast weather
```

**Skill** is 1 - MAE / MAE of the baseline: the share of the
baseline's error we removed.

The models (lessons 013 and 014, put to work):

- **Ridge: calendar + lags.** For each (Sunday, days ahead) pair, the
  target is log(rentals that day / last week's mean). Features:
  same weekday last week, Sunday's value, the trend over the last month
  (all relative to the level), day of week, holiday and working-day
  flags (known in advance), and yearly sine and cosine terms. One model
  covers all seven horizons, with the horizon as a feature. It already
  removes 16% of the baseline's error.
- **Perfect weather** is the weather that actually happened. We don't
  have it on Sunday, so it's **not allowed**: it's here to show how
  much weather *could* be worth, an upper bound (22%).
- **Forecast weather.** The dataset doesn't store old forecasts, so
  we **simulate** them: the real weather blurred more the further
  ahead you look (about 1.5 C of temperature error per square root of
  a day, and the wet/dry call wrong 4% of the time per day of lead).
  Ridge with forecast weather is no better than without, which is
  common: a lot of weather's value disappears once you have to predict
  it. Write this down; it's a finding.
- **Gradient boosting** with forecast weather does best of the honest
  models (855). Trees can use weather in a way the linear model
  can't, such as "rain matters a lot on a warm Saturday and little on
  a cold Tuesday".

The chosen model is picked by a rule written before seeing results:
lowest MAE on the selection weeks among the honest models.

## 3. Is the gain real?

```
  MAE gain per day vs seasonal naive, 95% CI         -> 236  [95, 377]
  weeks where the chosen model wins                  -> 24 of 35
  MAE by horizon (day 1..7)                          -> 814 847 861 655 840 1011 955
  big miss: 2012-04-22                               -> forecast 6411, actual 1027
  big miss: 2012-10-30                               -> forecast 5360, actual 1096
  big miss: 2012-04-18                               -> forecast 7994, actual 4367
```

- The gain is resampled **by week** (5,000 bootstrap draws of the 35
  weeks), because the days within a week share a forecast and their
  errors are correlated. 236 fewer rentals of error per day, and the
  interval stays well clear of zero.
- That interval is a little **optimistic**: we picked the winner on
  these same weeks, so some of its lead is luck (exercise 2). The
  sealed test is there to correct for that.
- Error grows with the horizon, but not smoothly: day 4 looks easy
  and days 6-7 look hard. With only 35 weeks, a few storms landing on
  particular weekdays decide that shape. Don't over-read per-horizon
  numbers from a short backtest.
- The **big misses** are days nobody could forecast from this data: a
  cold, very wet Sunday in April, and 30 October 2012, the day after
  **Hurricane Sandy** reached Washington (29 October had 22 rentals in
  the whole city). Looking at your worst errors is how you find out
  what the model can't know.

## 4. Intervals from past errors

```
  interval width at day 1 / day 7 (x forecast)       -> 0.70-1.08 / 0.77-1.30
  coverage on selection weeks (in-sample, ~80%)      -> 77%
```

The simplest honest interval: take the model's past errors on the
selection weeks, as ratios (actual / forecast), and use their 10th
and 90th percentiles for each horizon. No distributional assumption,
and it gets wider further out. It's a split-conformal interval in
spirit. The ranges are lopsided, stretching further below the
forecast than above at day 1: bad surprises (storms) are bigger than
good ones.

## 5. Open the sealed test, once

```
  seasonal naive                                     -> MAE 1474   skill +0%
  boosting: + forecast weather                       -> MAE 815   skill +45%
  ridge: + PERFECT weather (not allowed)             -> MAE 1014   skill +31%
  80% interval coverage on test                      -> 61%  (34 of 56 days)
  test MAE: Christmas week vs the rest               -> 1863 vs 666
  test bias (forecast - actual), mean                -> +178 rentals/day
```

Good news and bad news, both worth reporting.

- **The point forecast holds up.** MAE 815 on unseen weeks against
  1,474 for the baseline. The skill (45%) is even higher than in
  selection, but not because the model got better: the baseline got
  worse. Its first test week copies the Sandy week, and Thanksgiving
  and Christmas both break "same as last week". The model's own MAE
  (815) is close to its selection MAE (855), which is the reassuring
  number.
- **The intervals don't.** 61% coverage instead of 80%. Outside
  Christmas week it's 69%, still short. The selection weeks (March to
  October) had no big holidays and steady growth; November and
  December have holidays, early darkness and falling demand, and the
  model is **biased upwards** (+178 a day). Intervals built from one
  season's errors are too narrow for another season. That's the most
  useful thing the test told us.

What we must **not** do now: go back, tweak the model until the test
looks good, and report the new number. The test has been used. Fixes
get evaluated on new weeks.

## 6. The write-up

The script prints a draft. Here it is tidied into what you'd send:

> **Weekly rentals forecast: recommend using it for van planning,
> with holiday weeks planned by hand.**
>
> **What it does.** Every Sunday night it forecasts total rentals for
> each day of the coming week, with an 80% range.
>
> **How good is it.** On eight weeks it had never seen (November and
> December 2012), it was off by **815 rentals a day** on average,
> against 1,474 for "same as last week", **45% less error**. Over 35
> earlier weeks it beat that rule in 24.
>
> **Where it fails.** Storms and holidays. The worst days (Hurricane
> Sandy, Christmas week) were off by thousands, and the 80% ranges
> only held **61%** of the time in November and December, because
> they were built on spring and summer weeks. Over those two months it ran
> about 180 rentals a day too high.
>
> **What to do.** Use the forecast for normal weeks. For public
> holidays and named storms, override by hand. We'll rebuild the ranges
> with a full year of errors and check again after January.
>
> **What would change our mind.** If it loses to "same as last week"
> for three weeks running outside holidays, stop and investigate.

Same rules as in lesson 010: lead with the action, give the numbers in
units the reader cares about, say where it breaks, and leave a
tripwire.

## 7. The other two tracks: starters

If you choose B or C, the script has already run the baselines you'll
need to beat.

**Track B: recommend.** 400 simulated riders, 50 stations, 60 days of
trips. Train on days 0-44, test on 45-59 (a time split, as in lesson
015).

```
  riders / stations / train trips / test trips       -> 400 / 50 / 11056 / 3572
  recall@5: most popular stations                    -> 0.262
  recall@5: rider's own most-used stations           -> 0.484
  recall@5: item-item similarity                     -> 0.236
```

"Your own favourites" nearly doubles popularity, the return-trip
lesson from 017. Plain item-item similarity *loses* to popularity
here. Your job: find out why, and build something that beats 0.484
(matrix factorisation from 016, BPR from 017, or a blend with the
rider's own history), with an honest time split and a check for
popularity bias.

**Track C: connect.** In 1977, a karate club split in two after a
quarrel between its instructor (member 0) and its president (member
33).

```
  members / ties / homophily by final club           -> 34 / 78 / 86%
  label propagation from the 2 leaders: accuracy     -> 97%  (wrong: [8])
  link prediction, 8 hidden ties: Adamic-Adar AUC    -> 0.68  (only 16 pairs: very noisy)
```

Knowing only the two leaders, label propagation (lesson 019) gets 33
of 34 members right. Member 8 is the famous exception: almost every
method gets 8 wrong, because 8 has more ties to the president's side
but went with the instructor (Zachary's account has a personal reason:
8 was weeks away from a black-belt exam). Your job: try the methods
from lesson 019 and report which ones agree, then do link prediction
properly, with repeated random splits so the AUC isn't a coin flip on
16 pairs.

## How your write-up will be judged

Whichever track you pick, check your work against this list:

1. **A real question** from a named user, written down first.
2. **A plan before results**: metric, split, baseline, the rule for
   picking a model.
3. **An evaluation that can't cheat**: no future data, no test edges
   in your features, a test set opened once.
4. **A baseline that's hard to beat**, and an honest skill score
   against it.
5. **Uncertainty**: an interval, a bootstrap, or repeated splits.
6. **Failure analysis**: your worst cases, and why.
7. **One screen of write-up** that leads with the action.

## What phase 2 gave you

- **011-014:** time is not a column. Honest rolling-origin holdouts,
  autoregression, smoothing, and when features and ML actually help a
  forecast (section 2 above).
- **015-017:** recommenders are a ranking problem with traps:
  popularity, feedback loops, missing-not-at-random data, and
  return trips that beat clever models (track B).
- **018-019:** graphs as matrices, centralities, label propagation,
  graph convolution, embeddings and link prediction, and a check for
  homophily before you trust any of it (track C).
- **020:** put one of them together, evaluated honestly, and written
  up for someone who will act on it.

## What to do now

1. Run `lesson.py` and compare `output/forecast.png` with sections 5
   and 6.
2. Do the exercises in [`exercises.md`](exercises.md). The last one is
   your own track, end to end.
3. Phase 3 starts tomorrow with lesson 021: attention from first
   principles, the idea inside every Transformer, built in NumPy.
