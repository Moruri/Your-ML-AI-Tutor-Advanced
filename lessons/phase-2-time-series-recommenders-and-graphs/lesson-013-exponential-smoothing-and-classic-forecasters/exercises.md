# Lesson 013 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_013
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-013-exponential-smoothing-and-classic-forecasters/lesson.py my_work_013/
cd my_work_013
python lesson.py
```

---

## 1. Smoothing by hand

A small station's level is 100 rides. Using SES with alpha = 0.4, the
next three days bring 110, 90 and 120.

a) Update the level after each day.
b) What does SES forecast for each of the next seven days?
c) What weight does the 110 carry in the final level?

<details>
<summary>Check yourself</summary>

a) 0.4 x 110 + 0.6 x 100 = **104**; 0.4 x 90 + 0.6 x 104 = **98.4**;
0.4 x 120 + 0.6 x 98.4 = **107.04**.

b) **107.04 for all seven days.** SES forecasts a flat line at the
current level, whatever the horizon.

c) It entered with weight 0.4 and has been multiplied by 0.6 twice:
0.4 x 0.36 = **0.144**.

</details>

## 2. How far does a damped trend go?

A damped Holt model has slope +2 rides per day and phi = 0.9.

a) How much does the slope add to the forecast 3 days ahead?
b) What's the most it can ever add, however far ahead?
c) What would an undamped Holt add 30 days ahead?

<details>
<summary>Check yourself</summary>

a) 2 x (0.9 + 0.81 + 0.729) = **4.9 rides**.

b) 2 x 0.9 / (1 - 0.9) = **18 rides**. The geometric series
phi + phi^2 + ... adds up to phi / (1 - phi).

c) 2 x 30 = **60 rides**, and it keeps growing. That's the gap damping
was invented to close.

</details>

## 3. Which forecaster?

Pick SES, Holt (damped) or Holt-Winters for each, in one sentence.

a) Weekly spare-tyre orders for the whole fleet: no weekly pattern
(it's already weekly), no obvious growth.
b) Daily app sign-ups for a service that's growing every month.
c) Daily station rides, forecast for the next two weeks.

<details>
<summary>Check yourself</summary>

a) **SES.** No trend and no season to model, so a smoothed level is all
there is; extra states would only fit noise.

b) **Damped Holt**, possibly with a weekly season if sign-ups dip at
weekends. The damping keeps the growth from being extrapolated forever.

c) **Holt-Winters** (damped or not, it barely matters over 14 days),
and compare it with a calendar regression + AR on a rolling origin,
as in section 4.

</details>

## 4. Hands-on: tell Holt-Winters about holidays

Section 3 found Monday's seasonal number dragged down by Mashujaa Day.
Make `hw_filter` treat holidays as **missing**: compute the prediction
as usual, but on a public holiday replace the observed value with that
prediction before updating the states, so the holiday teaches the model
nothing. (Build a holiday flag from `dates` and `HOLIDAYS` before the
function, as section 4 does with `HOL`.)

Re-run. What happens to Monday, to the holdout and to the rolling
origin?

<details>
<summary>Check yourself</summary>

Add two lines after `err[i] = v - pred`:

```python
        if HOL[i]:
            v = pred
```

(and move the `HOL = ...` line above `hw_filter`).

```
  learned weekly pattern, Mon..Sun               -> +70 +87 +96 +73 +44 -121 -249
  011 holdout MAE: Holt-Winters, damped          -> 40.6
  Holt-Winters, damped                           -> MAE  47.5
  beats seasonal naive: HW, damped               -> 9 of 13 origins
```

Monday is back near its true +60. The rolling-origin MAE improves a
little (48.6 to 47.5) and the model wins one more origin, but the
single 28-day holdout gets *worse* (38.3 to 40.6). One window is noise;
13 windows are evidence. The fitted alpha and gamma also rise (about
0.40 and 0.44): with the holiday shocks gone, the optimiser is happy to
react faster.

</details>

## 5. Hands-on: drop the trend altogether

Add a fourth model to `SPECS`: `"HW, no trend": (False, True, False)`,
a level plus a weekly season, no slope. Before you run it, look again
at the damped model's fitted beta in section 2 and predict its rolling-
origin MAE.

<details>
<summary>Check yourself</summary>

```
  HW, no trend                                   -> MAE  48.6 | day 1 / 7 / 14:  47.9 /  61.2 /  57.3
```

Identical to Holt-Winters damped. Its optimiser had already set
beta = 0, which *is* the no-trend model: with beta = 0 the slope is
never updated, and damping shrinks its starting value by phi every
day until it's gone. Over 14 days, the season and the level do all
the work. That's not the same as saying the trend is unimportant: ask
either model for next November and it will be about 200 rides short.

</details>

## 6. Optional puzzle: Sunday's revenge

In section 4, SES's error is 117 on day 1 but **211** on day 7 and 187
on day 14. Without running anything, explain why those two days are
so bad, and why day 14 isn't simply worse than day 7.

<details>
<summary>Check yourself</summary>

The origins are Mondays, so days 7 and 14 are **Sundays**. SES
forecasts one flat number, close to an average day's rides, and Sunday
sits about 250 rides below an average day: it's the day the flat line
misses by most. Day 14 isn't worse than day 7 because SES's forecast
doesn't change with the horizon, so the error depends on *which
weekday* it is, not how far ahead. The model's real problem is the
missing season, not distance.

</details>
