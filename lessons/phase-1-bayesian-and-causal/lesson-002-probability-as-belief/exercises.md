# Lesson 002 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy (it writes an `output/` folder next to
wherever it sits). From the repo root:

```bash
mkdir -p my_work_002
cp lessons/phase-1-bayesian-and-causal/lesson-002-probability-as-belief/lesson.py my_work_002/
cd my_work_002
python lesson.py
```

---

## 1. Change the base rate

In section 1, suppose a fraud ring is active and 10% of top-ups are now
fraud. The flag is unchanged (catches 95%, false alarms on 5%). Before
computing: will P(fraud | flagged) roughly double, or more than that?

<details>
<summary>Check yourself</summary>

Far more than double. Per 10,000 top-ups: 1,000 fraud, 950 of them
flagged; 9,000 honest, 450 of them flagged. So
`950 / (950 + 450) = 0.679`, up from 0.161. The detector didn't
change at all; only the base rate did. That's why a flag's meaning
can shift under your feet when the world changes, and why monitoring
the base rate (lesson 080 of the first course) matters.

</details>

## 2. Why 0.333 and not 0.30?

With the flat prior, Station A's posterior mean is 0.333, not the
observed 0.30. Why is it pulled up?

<details>
<summary>Check yourself</summary>

The flat prior's mean is 0.5, and the posterior lands between the
prior and the data, so it's pulled a little towards 0.5. In fact it's
exactly `(3 + 1) / (10 + 2)`: a flat prior acts like one extra late
ride and one extra on-time ride added to your data. Lesson 003 shows
why that formula works, and gives you the same kind of shortcut for
the network prior.

</details>

## 3. Shuffle the rides

In section 4, would the final posterior change if the three late rides
had come first? What about the posterior *after ride 2*?

<details>
<summary>Check yourself</summary>

The final posterior is identical: it's the prior times seven
`(1 - theta)`s and three `theta`s, in any order. The posterior after
ride 2 does change, because by then it has seen different data (two
late instead of one). The order affects the journey, never the
destination.

</details>

## 4. Hands-on: how strong should the prior be?

The network prior `theta**1 * (1 - theta)**14` is a gentle one. In your
copy, add a much more confident prior with almost the same peak,
`theta**4 * (1 - theta)**44`, and rerun section 3 with it.

Predict: will Station A's posterior mean be above or below 0.185? Will
P(rate > 0.25) go up or down?

<details>
<summary>Check yourself</summary>

One way to write it:

```python
strong = normalise(theta**4 * (1 - theta) ** 44)
for name, prior in [("flat", flat), ("network", network), ("strong", strong)]:
```

The mean drops to about 0.133 and P(rate > 0.25) falls to about 0.01,
with a 95% interval of roughly [0.06, 0.23]. The stronger prior acts
like about 48 rides' worth of past evidence, so ten new rides barely
move it. That's fine *if* Station A really is like the others. If it's
on a steep hill, or next to a nightclub, a strong network prior hides
a real problem for a long time. A good prior is as strong as your
reasons for it, and no stronger.

</details>

## 5. Optional puzzle: twenty more rides

A month later, Station A has had 20 more rides, 6 of them late (still
30%). Starting from the network prior and all 30 rides (9 late), what
is the posterior mean now? Has the decision "worse than 25%?" moved?

<details>
<summary>Check yourself</summary>

Update the grid with `theta**9 * (1 - theta)**21`, or, equivalently,
update the 10-ride posterior with the 20 new rides (yesterday's
posterior, today's prior). The posterior mean rises to about 0.234,
with a 95% interval of roughly [0.13, 0.36], and P(rate > 0.25) climbs
to about 0.38, up from 0.18. The evidence is accumulating: the station
looks worse than typical, and the honest next step is a closer look
on the ground, not a closure. Thirty rides still isn't a thousand.

</details>
