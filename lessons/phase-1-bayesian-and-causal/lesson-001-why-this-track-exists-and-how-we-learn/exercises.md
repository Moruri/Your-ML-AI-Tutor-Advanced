# Lesson 001 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_001
cp lessons/phase-1-bayesian-and-causal/lesson-001-why-this-track-exists-and-how-we-learn/lesson.py my_work_001/
cd my_work_001
python lesson.py
```

---

## 1. Which question is it?

For each situation, say which of the three questions (how sure, what
caused what, is it worth it) matters most. (a) A new menu layout raised
orders 4% in its first two days. (b) Customers who use the loyalty app
spend twice as much, so marketing wants to push the app to everyone.
(c) A new model raises AUC from 0.91 to 0.92 but needs a GPU server.

<details>
<summary>Check yourself</summary>

(a) **How sure.** Two days is a small, possibly unusual sample; the
interval around 4% may well include zero. (b) **What caused what.**
Big spenders are more likely to install the app; installing may not
make anyone spend more. Only an experiment, or careful adjustment,
answers that. (c) **Is it worth it.** A 0.01 AUC gain may change very
few real decisions, and the server has a monthly cost. Count both.

</details>

## 2. Read the interval

Station A's 95% interval is [0.00, 0.60]. A colleague says "so there's
a 95% chance the true rate is between 0 and 0.6". Is that what a
bootstrap interval says?

<details>
<summary>Check yourself</summary>

Not quite. A bootstrap (or any frequentist) interval is a statement
about the *procedure*: intervals built this way cover the true rate
about 95% of the time. "There's a 95% chance the rate is in here" is a
statement of *belief* about this one rate, and it needs a prior. That
is exactly what a Bayesian credible interval gives you, and it's where
lesson 002 begins. With lots of data the two usually agree; with ten
rides they can differ a lot.

</details>

## 3. Why did the gap vanish?

In section 2, why does the observed gap of +0.203 shrink to about zero
inside each engagement group?

<details>
<summary>Check yourself</summary>

Because, within one group, engagement is held fixed. Whatever
difference was left between reminders-on and reminders-off riders came
from engagement, not from reminders. Engagement is a **confounder**: it
causes both the "treatment" and the outcome. Lesson 008 makes this
precise with causal graphs, including the cases where adjusting for a
variable makes things *worse*.

</details>

## 4. Hands-on: give the reminders a real effect

In your copy, change section 2 so reminders genuinely help: add 0.05 to
the stay probability of riders who get reminders, both in the
observational data and in the randomised experiment. (Hint: in the
experiment, the treatment is `assigned`, not `reminders`.)

Predict: what will the observed gap be, and what will the randomised
gap be?

<details>
<summary>Check yourself</summary>

One way to write it:

```python
stays = rng.random(n) < np.where(engaged, 0.85, 0.45) + 0.05 * reminders
...
stays_rct = rng.random(n) < np.where(engaged, 0.85, 0.45) + 0.05 * assigned
```

The randomised gap lands close to +0.05, the true effect. The observed
gap grows a little, to roughly +0.25, because it's the real effect
*plus* the confounding from engagement. The observational number is
still about five times too big. That's the usual situation in real
data: the effect isn't zero, it's just badly overstated.

</details>

## 5. Optional puzzle: where's the break-even?

In section 3, how much would a false alarm have to cost before the
simple rules and the big model are worth the *same*? Use the counts
from the output (79 / 200 and 83 / 578).

<details>
<summary>Check yourself</summary>

Set the two values equal: `79 * 2000 - 200 * c = 83 * 2000 - 578 * c`.
That gives `378 * c = 8000`, so `c` is about KES 21. If a false alarm
costs less than about KES 21, the big model is worth more; above that,
the simple rules win. In reality a false alarm that blocks someone's
money costs far more than KES 21, so the simple rules win easily. Once
you write the costs down, "which model is better?" often answers
itself.

</details>
