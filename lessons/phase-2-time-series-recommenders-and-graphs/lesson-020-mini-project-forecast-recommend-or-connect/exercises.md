# Lesson 020 - Exercises

Two quick checks, two hands-on experiments, and then the real work:
your own track, end to end. Predict first, then run.

Activate your venv. For the hands-on tasks, copy the lesson folder
somewhere of your own (the script needs `day.csv` next to it). From the
repo root:

```bash
mkdir -p my_work_020
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-020-mini-project-forecast-recommend-or-connect/{lesson.py,day.csv} my_work_020/
cd my_work_020
python lesson.py
```

---

## 1. Spot the leak

A colleague proposes extra features for the Sunday-night forecast.
For each, say whether it's allowed, and why.

a) Whether the target day is a public holiday.
b) The `registered` count on the target day.
c) The temperature actually recorded on the target day.
d) The weather service's forecast for the target day, as issued on
   Sunday.
e) The mean of `cnt` over the whole of 2012, used to scale the series.

<details>
<summary>Check yourself</summary>

a) **Allowed.** Holidays are known years in advance.

b) **Leak.** `cnt = casual + registered`, so it hands over most of the
answer. Any column measured on the target day is suspect.

c) **Leak** for this question. It's the "PERFECT weather" model: fine
as an upper bound, never as a result you report.

d) **Allowed**, and it's what section 2 simulates. If you can get
archived forecasts, use them instead of the simulation.

e) **Leak.** A 2012 mean includes the future of every 2012 origin.
Scale with information up to the origin only, as the script does with
last week's level.

</details>

## 2. The winner's curse

On the selection weeks we tried five honest-ish models and kept the
best (855). On the test it scored 815, but the gap between the chosen
model and the runner-up was small.

a) Why is the best of five validation scores biased downwards (too
   good), even if no model is better than another?
b) Why didn't it show up here as a worse test score?
c) What would you do if you had compared 200 feature sets on the
   selection weeks?

<details>
<summary>Check yourself</summary>

a) Each validation MAE is a true error plus noise. Taking the minimum
picks the model whose noise happened to be most favourable, so the
minimum is expected to be lower than that model's true error.

b) The bias is there, but small with five candidates, and it's swamped
by the change of season: a test MAE depends on how hard the test
weeks are. Compare models on the *same* weeks (skill against the
baseline, or the gap between models), not raw numbers across periods.

c) Keep a third slice: choose on selection weeks, confirm on a
validation slice, and only then open the test. Or prefer a simpler
model when the difference is within the bootstrap noise of section 3.
The more you compare, the more your best score flatters you.

</details>

## 3. Hands-on: wider ranges

The 80% ranges only covered 61% of test days. In your copy, change
the percentiles in section 4 from 10 and 90 to 5 and 95, and rerun.

a) Predict the new test coverage. Will it reach 80%?
b) What does the range look like for day 1 now?
c) Is this a fix?

<details>
<summary>Check yourself</summary>

    nominal 90% range (5th-95th percentile):
    test coverage                -> 75% overall, 86% outside Christmas week
    day-1 range (x forecast)     -> 0.50 to 1.12

a) Better, but a *nominal 90%* range now covers only 75% of test
days. Outside Christmas week it's 86%, close to what an 80% range
should do.

b) Very lopsided: down to half the forecast, up only 12%. The lower
tail is set by storm days.

c) It's a **patch**, chosen after looking at the test, so the 75% is
no longer an honest test result. The real fixes are to build the
ranges from errors that cover every season (a full year of rolling
origins), to widen the ranges for holiday weeks on purpose, or to
make the interval adapt as recent errors grow (adaptive conformal
methods do exactly this). Evaluate any of these on new weeks.

</details>

## 4. Hands-on: two kinds of rider

Section 1 found almost no weekly pattern in total rentals. Look
instead at the two columns that make up the total:

    casual      mean by weekday (Sun..Sat): 1035  548  424  368  401  520 1050
    registered  mean by weekday (Sun..Sat): 2260 2854 3091 2903 3040 3001 2308

(both from before March 2012). Casual riders peak at weekends,
registered riders (mostly commuters) on weekdays. Added up, the two
patterns almost cancel.

In your copy, forecast each one with the chosen model and add the two
forecasts. The quickest way: wrap sections 2-5 in a function, replace
`cnt[:]` with `col("casual")` for one run and `col("registered")` for
another, keep the two prediction arrays, then restore `cnt` and score
their sum against the real totals.

Predict: does forecasting the parts beat forecasting the total?

<details>
<summary>Check yourself</summary>

    boosting + forecast weather       selection MAE   test MAE
    on the total (the lesson)         855             815
    casual + registered, summed       827             575

A small gain on the selection weeks and a big one on the test (29%
less error). Casual riders respond to weekends, holidays and weather
in a completely different way from commuters, and one model on the
sum has to learn a muddle of both. Over Thanksgiving and Christmas,
commuting collapses while leisure riding doesn't, which is exactly
when the total-only model went wrong.

Two cautions. This is a second look at the sealed test, so the 575 is
now an optimistic number; the gain on the selection weeks is the
honest evidence, and it's much smaller. And splitting is a general
idea, not a trick: when a total is made of groups that behave
differently, model the groups (lesson 005's partial pooling is the
same thought).

</details>

## 5. Optional: one more honest baseline

Add a third baseline: the **mean of the same weekday over the last
four weeks** (`np.mean([cnt[t + h - 7 * k] for k in 1..4])`, using
only days up to t). Before running it, predict whether it beats
seasonal naive, and whether it changes the story of the write-up.

<details>
<summary>Check yourself</summary>

                      selection MAE   test MAE
    seasonal naive    1090            1474
    four-week mean     922            1292
    chosen model       855             815

Averaging four weeks cancels a lot of one-off noise (a single rainy
Saturday no longer sets next Saturday's forecast), so it beats seasonal
naive clearly. In the first test week, where seasonal naive copied the
Sandy days, it's off by 955 a day instead of 2,045.

And it changes the story. On the selection weeks the free four-week
average (922) is level with the ridge model (917), so ridge earned
nothing. The chosen boosting model still wins (skill 7% on selection,
37% on test against the four-week mean), but the honest headline is
"37% better than a four-week average", not "45% better than last
week". A model has to earn its complexity against the strongest simple
thing, not the weakest baseline you could find.

</details>

## 6. The deliverable: your own track

Choose one track and take it to a one-screen write-up, using the
checklist at the end of the README (real question, plan first, honest
evaluation, strong baseline, uncertainty, failure analysis, one
screen).

- **A. Forecast**, going further: forecast `casual` and `registered`
  properly (decided on the selection weeks only), or move to the
  hourly file of the same dataset (`hour.csv` from the UCI "Bike
  Sharing" page) and forecast tomorrow's 24 hours. Report skill
  against seasonal naive *and* the four-week average.
- **B. Recommend**: beat 0.484 recall@5 on the simulated riders with
  something from lessons 016-017. Explain first why item-item
  similarity lost to popularity (hint: what does it recommend for a
  rider whose favourite station is also everyone's favourite?). If
  you want real data, MovieLens 100K (from GroupLens) is small and
  public; split it by time, not at random.
- **C. Connect**: on the karate club, compare label propagation, SGC
  and a walk embedding from lesson 019, starting from the two leaders
  only. Then do link prediction with 50 random splits and report the
  mean AUC and its spread for common neighbours, Adamic-Adar and
  embedding cosine. Which method's errors include member 8, and why?

<details>
<summary>What a good write-up looks like</summary>

It fits on one screen and opens with the decision, in the reader's
words ("use it for normal weeks; plan holidays by hand"). The numbers
are in the reader's units (rentals a day, share of riders who found
their next station in the top five, number of members placed in the
right group) and each has a range. It names the baseline and the skill
against it. It says where the method breaks, with an example. It ends
with what would make you change your mind.

What it leaves out: the model zoo, hyperparameters, and every number
you computed along the way. Those go in an appendix for whoever asks.

If your result is "the simple baseline wins", that's a perfectly good
write-up. Saying so, with evidence, saves someone from shipping
complexity that doesn't pay.

</details>
