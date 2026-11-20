# Lesson 012 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_012
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-012-autoregression-stationarity-and-holdouts-that-dont-cheat/lesson.py my_work_012/
cd my_work_012
python lesson.py
```

---

## 1. An AR forecast by hand

A station's calendar regression predicts 300 rides for each of the
next three days. Its residual follows an AR(1) with no constant and
phi = 0.6. Today's residual was +50 (a busy day).

a) Forecast the residual and the rides for the next three days.
b) What does the forecast look like 30 days ahead, and why?
c) Same question if phi were 1.0.

<details>
<summary>Check yourself</summary>

a) Residuals: 0.6 x 50 = **+30**, then 0.6 x 30 = **+18**, then
0.6 x 18 = **+10.8**. Rides: **330, 318, 310.8**.

b) 50 x 0.6^30 is about 0.00001, so the forecast is just the
regression's **300**. A stationary AR's memory fades geometrically, and
far ahead the forecast is the long-run mean.

c) With phi = 1.0 the residual forecast stays at **+50 forever**: a
random walk's best guess is "wherever it is now", and the uncertainty
around that guess keeps growing (section 1's spreading walks).

</details>

## 2. Stationary or not?

Say whether each is likely stationary, and if not, what you'd do
before fitting an AR.

a) Daily temperature in Nairobi over five years.
b) The total number of bikes the operator has ever bought, by month.
c) The daily share of rides that start before 9am.

<details>
<summary>Check yourself</summary>

a) **Not stationary as-is**, because the mean depends on the time of
year. But the yearly pattern is predictable, so regress out a seasonal
pattern (or subtract each day's long-run average) and the remainder is
probably stationary.

b) **Not stationary**: a cumulative count only goes up, much like a
random walk with drift. Difference it (bikes bought *this* month), and
that series may well be stationary.

c) **Probably stationary**, or close to it: a share is bounded between
0 and 1 and tends to hover around a level, perhaps with a weekly
pattern you'd remove first. Check with a plot and the ADF test, and
watch for a slow drift if commuting habits change.

</details>

## 3. Why not -1.64?

In two sentences: why can't the ADF test use the usual normal critical
value, and what goes wrong if it does?

<details>
<summary>Check yourself</summary>

Under the null hypothesis the series is a random walk, and regressing
a random walk on its own past breaks the assumptions behind the usual
t-distribution, so the t-statistic has its own skewed Dickey-Fuller
distribution, centred well below zero. With -1.64 you'd reject far too
often: in section 2, 46% of real random walks were called stationary,
not 5%.

</details>

## 4. Hands-on: choose p inside each origin

Section 5 used *p* = 6, chosen once with data that came after most of
the origins. Fix the leak: in the rolling-origin loop, refit the
calendar regression at each origin, pick *p* on its residual with
`pick_p(res, 14)`, and forecast with that *p*. Print the chosen orders
and the overall MAE.

Then try a fixed AR(1) for comparison.

<details>
<summary>Check yourself</summary>

```python
ps, E = [], []
for o in origins:
    hist = y[:o]
    _, _, res = ols(calendar(0, o), hist)
    p = pick_p(res, 14)
    ps.append(p)
    E.append(np.abs(y[o:o + HZ] - fc_reg_ar(hist, HZ, p)))
print(ps, np.mean(E))
print(np.mean([np.abs(y[o:o + HZ] - fc_reg_ar(y[:o], HZ, 1)) for o in origins]))
```

The orders move around (`[5, 5, 12, 7, 7, 7, 5, 6, 6, 6, 6, 6, 6]`),
but the MAE is **40.2**, the same as before to one decimal. A plain
AR(1) gets **40.9**. So here the leak was harmless and the exact order
hardly matters; most of the gain comes from yesterday. That's a common
outcome, but you only know it's harmless after you've done the honest
version.

</details>

## 5. Hands-on: more origins

Change `STEP` to 7 and `N_ORIG` to 26: an origin every week from 26 May,
each still forecasting 14 days (so the windows overlap). Do the
rankings hold?

<details>
<summary>Check yourself</summary>

```
  seasonal naive                                 -> MAE   55.2
  degree-8 curve                                 -> MAE  121.3
  weekly difference + AR(21)                     -> MAE   47.9
  calendar regression only                       -> MAE   41.8
  calendar regression + AR(6)                    -> MAE   38.9
```

(Only the overall MAE shown.) Same order, similar gaps. With
overlapping windows the 26 results aren't independent, so this isn't
twice the evidence, but the averages are smoother and you can trust
the ranking more. The figure's horizon curves get less bumpy too.

</details>

## 6. Optional puzzle: the holiday echo

The seasonal naive's error on day 7 at the first origin is **350**
rides, against an average of about 58 at the other 12 origins. Without
running anything, explain why. Then suggest a one-line change to the
seasonal naive that would fix it, and say which lesson's tool it
borrows.

<details>
<summary>Check yourself</summary>

The first origin is Monday 2 June 2025, so "last week" includes Sunday
1 June, Madaraka Day. The seasonal naive copies that holiday's dip
(about 300 rides) onto the next two Sundays, which were ordinary.

A fix: before copying, replace any holiday in last week with the same
weekday from the week before (or add back the holiday effect estimated
by the calendar regression). Either way you're using **known calendar
information**, the same idea as the holiday flag in section 2(b), and
the start of what lesson 014 does with exogenous features.

</details>
