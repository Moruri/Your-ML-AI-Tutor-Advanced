# Lesson 008 - Exercises

Three quick checks, a hands-on experiment, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on task, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_008
cp lessons/phase-1-bayesian-and-causal/lesson-008-confounding-graphs-and-what-to-adjust-for/lesson.py my_work_008/
cd my_work_008
python lesson.py
```

---

## 1. Name the role

Back to the reminder texts from lessons 006-007, this time with
observational data. We want the effect of a text (`T`) on returning
late (`Y`). For each variable, say whether it's a confounder, mediator,
collider, or none of these, and whether you'd adjust for it.

a) `past_late`: operations texted riders with more past late rides, and
those riders are late more often anyway.
b) `opened_app`: the text makes riders open the app, and opening the
app (to check the return time) makes them return on time.
c) `got_fine`: riders get a fine if they're late, and the text
mentions fines, so texted riders who are late notice and dispute them,
which is how a fine gets logged.
d) `phone_model`: affects nothing here.

<details>
<summary>Check yourself</summary>

a) **Confounder** (`T <- past_late -> Y`). Adjust.
b) **Mediator** (`T -> opened_app -> Y`). Don't adjust if you want the
total effect of texting; holding app opens fixed removes the main way
the text works.
c) **Collider** (`T -> got_fine <- Y`). Don't adjust, and don't filter
to "riders with a logged fine".
d) Neither. Adjusting does no harm in theory, but it adds noise and
gains nothing.

</details>

## 2. "We controlled for everything"

A report says: "We regressed duration on e-bike use, controlling for
every column in the rides table (distance, hill, rain, promo, speed,
flagged), and found e-bikes save only about a minute." Using sections 3-5,
what went wrong?

<details>
<summary>Check yourself</summary>

They adjusted for `speed` (a mediator) and `flagged` (a collider, and
also downstream of the treatment). Controlling for speed alone already
drives the estimate to about zero (+0.08 with `{dist, hill, speed}`);
the collider then drags it negative again. Run
`effect(["dist", "hill", "rain", "promo", "speed", "flagged"])`: about
-1.3 minutes, a third of the truth, made of two biases that happen to
partly cancel. "Adjust for everything" is not a safe default; **adjust
for the right things**.

</details>

## 3. Survivorship as a collider

During a war, analysts studied bullet holes on bombers that came
**back** from missions, and proposed armouring the spots with the most
holes. Explain the problem using the words "collider" or "selection".

<details>
<summary>Check yourself</summary>

Whether a plane returns depends on where it was hit (hits to the
engine bring planes down). Studying only returners is conditioning on
"returned", a consequence of the hit location. Among survivors, engine
hits are rare not because they don't happen but because those planes
didn't come back. The fix (armour where returners *aren't* hit) comes
from reasoning about the selection, exactly the flagged-rides trap in
section 5.

</details>

## 4. Hands-on: an unmeasured confounder

Add a variable nobody records: `rush`, "running late for work".
Rushed riders are likelier to grab an e-bike **and** ride faster
anyway. In your copy:

1. Add `"rush": []` to `PARENTS`, add `"rush"` to the parents of both
   `ebike` and `speed`, and give it a spot in the figure:
   `POS["rush"] = (0, 0.5)` and `ROLE["rush"] = "black"`.
2. Simulate it after `promo`: `rush = (rng.random(n) < 0.3).astype(float)`,
   add `+ 2.0 * rush` inside the e-bike `expit(...)`, add `+ 6 * rush`
   to `speed`, and put `rush=rush` in `DATA`.
3. Rerun. Then try `effect(["dist", "hill", "rush"])`.

Predict first: does `{dist, hill}` still get -4?

<details>
<summary>Check yourself</summary>

No. `{dist, hill}` now gives about **-4.7**, and the verdict says why:
`no: open ebike <- rush -> speed -> duration`. Rushed riders are both
on e-bikes and fast, so e-bikes get credit for the rush. No set of the
*recorded* variables is valid. If you could record `rush`,
`{dist, hill, rush}` gives about -3.9 and passes the check.

Also look at `{dist, hill, promo}`: about **-4.9**, *worse* than
without promo. With an unmeasured confounder, adjusting for a
treatment-only variable removes the "clean" variation in e-bike use and
leaves a bigger share driven by `rush`, so it amplifies the bias. One
more reason to keep treatment-only variables out.

</details>

## 5. Optional puzzle: find every valid set

Write a few lines that try every subset of `["dist", "hill", "rain",
"promo", "speed", "flagged"]` (use `itertools.combinations`) and print
the ones where `verdict(Z) == "yes"`. Before running: how many do you
expect, and what's the smallest?

<details>
<summary>Check yourself</summary>

```python
from itertools import combinations
cands = ["dist", "hill", "rain", "promo", "speed", "flagged"]
for k in range(len(cands) + 1):
    for Z in combinations(cands, k):
        if verdict(list(Z)) == "yes":
            print(Z)
```

Four: `{dist, hill}`, plus that with `rain`, with `promo`, and with
both. The smallest is `{dist, hill}`, and every valid set must contain
it, because each of those two variables sits on a backdoor that nothing
else blocks. Among the four, section 6 says to prefer
`{dist, hill, rain}`: valid *and* the tightest interval. Real tools
such as DAGitty do exactly this search on bigger graphs.

</details>
