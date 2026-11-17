# Lesson 006 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_006
cp lessons/phase-1-bayesian-and-causal/lesson-006-potential-outcomes-and-causal-questions/lesson.py my_work_006/
cd my_work_006
python lesson.py
```

---

## 1. Name the counterfactual

For each claim, write down `Y0`, `Y1`, and which of them was never
observed.

a) "Riders who used the new app returned on time 92% of the time."
b) "After we cut prices at Station C, rides went up 30%."
c) "Patients who took the drug recovered faster than those who didn't."

<details>
<summary>Check yourself</summary>

a) `Y1` = on time with the app, `Y0` = on time without it. For app
users we never see `Y0`. The 92% says nothing about the effect, since
keen, organised riders may be the ones who installed the app.
b) `Y1` = rides at C with the price cut, `Y0` = rides at C over the same
weeks without it. `Y0` is missing; maybe the weather improved or term
started. A before/after comparison assumes nothing else changed (lesson
009 does this properly).
c) `Y1` = recovery time on the drug, `Y0` = without. Each patient shows
only one. If doctors gave the drug to milder cases, that's selection
bias, exactly as in section 3.

</details>

## 2. ATE or ATT?

Operations asks two separate questions: (a) "Should we stop the texts
we're already sending?" and (b) "Should we roll texts out to every
rider?" Which average effect answers each?

<details>
<summary>Check yourself</summary>

(a) is about the riders currently being texted, so it's the **ATT**,
`E[Y1 - Y0 | texted]` (about -0.05 here). (b) is about everyone, so it's
the **ATE** (also about -0.05 in this simulation, though the two needn't
match). They differ whenever the effect varies and the treated group
isn't typical, which is most of the time. Always ask "effect on whom?"

</details>

## 3. Is the naive comparison always too high?

In section 3 selection bias made texts look harmful. Describe a
targeting rule under which the naive comparison would make texts look
**better** than they really are.

<details>
<summary>Check yourself</summary>

Text the riders who are *least* likely to be late, say those with zero
past late rides. Then `E[Y0 | T=1]` is below `E[Y0 | T=0]`, the
selection bias is negative, and it adds to the real (negative) effect:
texts look more powerful than they are. The sign of the bias depends on
how treatment was assigned, which is why you can't just "correct in
your head".

</details>

## 4. Hands-on: estimate uplift from an experiment

In real life you can't sort by `true_uplift`. But section 3's coin-flip
data (`t_r`, `y_r`) is a randomised experiment, and that's enough to
estimate it. In your copy:

1. Build features `Xr` with `1, past_late, dist, past_late**2, past_late * dist`.
2. Fit one logistic model on the riders with `t_r == 0` and one on
   `t_r == 1` (`fit_logistic`).
3. Estimated uplift = predicted lateness without minus with.
4. Text the top 4,000 by estimated uplift and count late returns
   prevented with `late_after`.

Predict first: will it get near the oracle's 413?

<details>
<summary>Check yourself</summary>

```python
Xr = np.column_stack([np.ones(n), past_late, dist, past_late**2, past_late * dist])
b0 = fit_logistic(Xr[t_r == 0], y_r[t_r == 0])
b1 = fit_logistic(Xr[t_r == 1], y_r[t_r == 1])
est_uplift = expit(Xr @ b0) - expit(Xr @ b1)
pick = np.argsort(-est_uplift)[:budget]
print(base - late_after(pick), np.corrcoef(est_uplift, true_uplift)[0, 1])
```

About 310 prevented: well short of the oracle's 413, but far better
than the risk model's 134 and the random 198. The estimated uplift
correlates with the truth at roughly 0.7. Try it with only the three
linear features and it drops to about 260, because the effect peaks in
the middle of `past_late` and a straight line can't bend. This
"two-model" approach (often called a **T-learner**) is the simplest
uplift model. Phase 3 and 4 tools do it better, but every one of them
needs data where treatment was randomised, or carefully adjusted for.

</details>

## 5. Optional puzzle: sleeping dogs

Suppose a text annoys riders who are *never* late (0 past late rides),
and 5% of them start returning late out of spite. Change the simulation
so `Y1` can be 1 when `Y0` is 0 for those riders. What happens to the
ATE, and which policy suffers most?

<details>
<summary>Check yourself</summary>

One way, right after `y1` is created:

```python
spite = (past_late == 0) & (y0 == 0) & (rng.random(n) < 0.05)
y1 = np.where(spite, 1, y1)
```

(Recompute `ite` and `true_uplift` after this; for `true_uplift`,
subtract 0.05 from those riders' `p0 - p1`.) The ATE moves a little
towards zero, because zero-history riders are a large share of
everyone. Random targeting and blanket texting now give back some of their gains,
while the risk-based list barely notices, since it never picks those
riders. The uplift ordering simply puts them at the bottom. These are
the **sleeping dogs**: the action hurts them, and the only way to find
them is to look at effects, not outcomes.

</details>
