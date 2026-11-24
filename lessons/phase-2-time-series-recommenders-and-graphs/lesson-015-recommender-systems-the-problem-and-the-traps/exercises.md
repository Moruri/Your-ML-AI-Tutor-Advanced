# Lesson 015 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_015
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-015-recommender-systems-the-problem-and-the-traps/lesson.py my_work_015/
cd my_work_015
python lesson.py
```

---

## 1. Score a list by hand

A rider gets a list of 4 places. They later visit the places at
positions **2** and **4**, plus one place that wasn't on the list.

a) Precision@4 and recall@4?
b) DCG@4 and NDCG@4? (log2 of 2, 3, 4, 5 is 1, 1.585, 2, 2.322.)
c) You swap the list so the hits sit at positions 1 and 2. Which of
   the three numbers change?

<details>
<summary>Check yourself</summary>

a) Precision@4 = 2/4 = **0.50**. Recall@4 = 2/3 = **0.67** (three
future visits, two caught).

b) DCG = 1/1.585 + 1/2.322 = 0.631 + 0.431 = **1.062**. The ideal puts
three hits at positions 1-3: 1 + 0.631 + 0.5 = 2.131. NDCG = 1.062 /
2.131 = **0.498**.

c) Only **NDCG** changes: DCG becomes 1 + 0.631 = 1.631 and NDCG
0.765. Precision and recall don't care about order inside the list.

</details>

## 2. Which metric?

Pick the metric that best matches each worry:

a) "Riders only ever look at the first two slots."
b) "Half our partner places have never appeared in anyone's tab."
c) "Lots of riders open Explore and find nothing at all for them."
d) "Of everything a rider eventually does, how much did we suggest?"

<details>
<summary>Check yourself</summary>

a) **NDCG@k** with a small k (or precision@2): position matters.

b) **Coverage**. No per-rider metric can see this; a perfect-looking
popularity list can have 13% coverage.

c) **Hit rate@k**: the share of riders with at least one right answer.

d) **Recall@k**.

</details>

## 3. Spot the trap

A teammate reports: "Our new model gets NDCG@10 of 0.31; popularity
only gets 0.22." Their notebook held out a random 20% of visits and
ranked all 240 places for each rider, including places the rider had
already been to in training. Name two reasons to doubt the comparison.

<details>
<summary>Check yourself</summary>

1. **Random split.** Section 5 showed a random split flatters even
popularity by about 60%, because counts include the future. A model
that learns from co-visits gets the same leak, often more. Re-run both
on a time split.

2. **Re-recommending known places.** If the model can put places the
rider already visited at the top, either those count as misses (an
unfair penalty, handled differently for each model) or, with revisits
allowed, the "prediction" is just "you'll go back to your usual café",
which isn't what the Explore tab is for. Decide the rule, then apply it
to every model the same way (the lesson removes them, in `ranked`).

A third, softer one: 0.22 is the *random-split* popularity number from
this lesson, so check that both numbers really came from the same run.

</details>

## 4. Hands-on: change k

In your copy, evaluate "most popular" and "your genres x popularity"
at k = 5, 10 and 20 (call `metrics(..., k)`). Predict first: which
metrics go up with k, which go down?

<details>
<summary>Check yourself</summary>

What we got:

| k | model | P@k | R@k | hit | NDCG@k | cover |
|---|---|---|---|---|---|---|
| 5 | most popular | 0.059 | 0.131 | 26% | 0.108 | 8% |
| 5 | genres x popularity | 0.087 | 0.191 | 36% | 0.160 | 44% |
| 20 | most popular | 0.037 | 0.310 | 52% | 0.169 | 20% |
| 20 | genres x popularity | 0.049 | 0.402 | 62% | 0.233 | 73% |

Longer lists catch more (recall, hit rate and coverage rise) but each
slot is less likely to be right (precision falls). NDCG@k rises too,
because it's normalised per rider and extra slots can only add hits.
That's why you **never compare metrics at different k**, and why you
should pick k to match the screen: the Explore tab shows ten.

The ranking of the two models doesn't change with k here. When it does,
that tells you one model is better at the very top and the other
further down.

</details>

## 5. Hands-on: how much shrinkage?

In section 6, try m = 0, 1, 3, 10, 30, 100 and 1000 in the shrunk
average, and print `top_true_rank` for each. (m = 0 is the raw
average.)

<details>
<summary>Check yourself</summary>

| m | mean true rank of the top 10 | median ratings each |
|---|---|---|
| 0 | 72.6 | 7 |
| 1 | 33.6 | 36.5 |
| 3 | 22.2 | 37.5 |
| 10 | 6.9 | 60 |
| 30 | 6.9 | 60 |
| 100 | 6.2 | 60 |
| 1000 | 5.8 | 61.5 |

Even m = 1 halves the damage, and from about 10 on the list is stable.
A good rule of thumb is to set m near the typical number of ratings a
place has, or to estimate it from the data, as lesson 005 did for the
stations' prior.

But look at m = 1000: still improving. With a huge m, the score is
roughly the city mean plus (sum of stars above the mean) / 1000, which
rewards places with **many** good ratings. Here that works, because
visits and quality go together. In a city where heavily advertised
mediocre places get lots of visits, it wouldn't. That's the puzzle
below.

</details>

## 6. Optional puzzle: why did the oracle lose slots?

The oracle ranks by each rider's true enjoyment, yet its precision@10
is only 0.083 and its coverage is 85%, much higher than any popularity
baseline. Using how the simulation draws visits, explain (a) why the
oracle can't reach high precision even in principle, and (b) one way
to build a scorer that predicts *visits* better than the oracle does,
using only things the oracle doesn't know.

<details>
<summary>Check yourself</summary>

a) Visits are **random draws**: a rider with 2-3 visits in the test
window picks them from many places they'd enjoy, so most of the ten
places they'd like most simply don't get drawn this time. Also, visit
probability is enjoyment **times visibility times new-place buzz**,
and the oracle ignores the last two. It recommends great places nobody
has heard of (hence the high coverage), and riders keep going to the
well-known ones.

b) Multiply in what drives visits. We tried two, keeping the oracle's
enjoyment and adding one thing it ignores:

| scorer | P@10 | NDCG@10 | cover |
|---|---|---|---|
| oracle: enjoyment only | 0.083 | 0.234 | 85% |
| exp(2 x enjoyment) x (training visits + 1) | 0.085 | 0.250 | 58% |
| exp(2 x enjoyment) x true visibility | 0.105 | 0.321 | 76% |

The visit count is a noisy stand-in for visibility and only just helps;
the true visibility helps a lot. Both predict *visits* better than pure
enjoyment, because the target is visits, not delight. Whether you
*should* optimise for that is a product question: a tab that mostly
predicts the obvious, well-signposted place is accurate and not very
useful. Lesson 017 makes the choice of target explicit.

</details>
