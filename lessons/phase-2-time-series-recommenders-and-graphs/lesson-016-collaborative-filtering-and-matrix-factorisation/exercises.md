# Lesson 016 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_016
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-016-collaborative-filtering-and-matrix-factorisation/lesson.py my_work_016/
cd my_work_016
python lesson.py
```

---

## 1. One prediction by hand

A k = 2 model has mu = 3.4. Rider Wanjiru has bias +0.3 and vector
(1.0, -0.5). Karura Café has bias -0.2 and vector (0.8, 0.4).

a) Predicted stars?
b) Which single number would you change to make the café a worse
   match for her taste, without changing how good it is in general?

<details>
<summary>Check yourself</summary>

a) Dot product: 1.0 x 0.8 + (-0.5) x 0.4 = 0.8 - 0.2 = 0.6. Prediction
= 3.4 + 0.3 - 0.2 + 0.6 = **4.1 stars**.

b) Any coordinate of the café's **vector** (or hers): for example make
its second coordinate larger, since she has a negative weight there.
The **bias** is "how good it is for everyone", so leave that alone.
The split between bias and vector is exactly the split between
"general quality" and "taste match".

</details>

## 2. Why alternate?

a) Why is each half-step of ALS easy, when the whole problem isn't?
b) In one round of the lesson's ALS, how many small linear systems are
   solved, at most, and how big is each one?
c) Name one thing ALS makes easy that a single big optimiser wouldn't.

<details>
<summary>Check yourself</summary>

a) With the place vectors fixed, the prediction is **linear** in the
rider's unknowns (p_u and b_u), so minimising squared error plus a
ridge penalty is an ordinary ridge regression with a closed-form
answer. With both sides unknown, the product p_u . q_i makes the
problem non-convex.

b) One per rider and one per place that has ratings: at most 1,200 +
240 = 1,440 systems, each (k + 1) x (k + 1), so 3 x 3 for k = 2. Tiny.

c) **Folding in** (section 6): a new rider is just one more half-step.
Also, every rider's solve is independent of every other rider's, so
ALS parallelises trivially, which is why it's the classic method for
very large matrices.

</details>

## 3. Read the grid

From section 4's grid: a colleague says "k = 8 is useless, the factors
are noise". Using section 5's numbers, give a fairer one-sentence
verdict.

<details>
<summary>Check yourself</summary>

Something like: "k = 8's factors carry more real structure (about half
of each genre's variation, against 12-27% for k = 2), but with a median
of 12 ratings per rider, eight numbers per rider can't be estimated
precisely, so its predictions are worse (0.929 against 0.901 on
validation)." With more ratings per rider, the balance would move
towards bigger k.

</details>

## 4. Hands-on: shrink the similarities

In section 3, the similarity between two places is multiplied by
n / (n + 10), where n is the number of riders who rated both. Refit
`item_knn(tr, k=20, shrink=...)` with shrink = 0, 10 and 100, and
print the validation RMSE and the genres of place 123's five nearest
neighbours. (With shrink = 0 you'll see a divide warning: places with
no shared riders give 0 / 0. That's expected; those pairs just never
become neighbours.)

<details>
<summary>Check yourself</summary>

What we got:

| shrink | validation RMSE | place 123's 5 nearest |
|---|---|---|
| 0 | 0.987 | nature, nightlife, nature, food, culture |
| 10 | 0.957 | culture, nature, food, nature, nature |
| 100 | 0.954 | nature, nature, nature, culture, nature |

Without shrinkage, a pair of places with two shared riders who happen
to agree gets a cosine near 1 and dominates the neighbour list; the
neighbours become random-looking and the error is *worse than biases
alone* (0.979 on test). Shrinking fixes that. Heavy shrinkage
effectively ranks neighbours by "similar **and** well-supported",
which here is slightly better still. (These neighbours come from the
training window only, so they differ from the lesson's refitted list.)

</details>

## 5. Hands-on: the fold-in penalty

In section 6, the new rider's ridge solve uses the model's lambda (3).
Try lambda = 0.3, 3 and 30 for the fold-in only (keep the place
vectors), with n = 1, 3 and 10 known ratings.

<details>
<summary>Check yourself</summary>

| fold-in lambda | n = 1 | n = 3 | n = 10 |
|---|---|---|---|
| 0.3 | 1.052 | 1.056 | 0.893 |
| 3 | 1.011 | 0.958 | 0.870 |
| 30 | 1.033 | 1.009 | 0.956 |

A tiny penalty lets one or three ratings swing the vector wildly, and
at n = 1 and 3 it's worse than knowing nothing about the rider (1.044
at n = 0). A huge penalty barely moves away from zero, so it wastes the
ratings, and at n = 10 it's no better than biases only (0.956). The
training lambda is about right, because it was chosen for riders with
similar amounts of data. In a real system you might use a slightly
larger penalty for brand-new users and relax it as they rate more.

</details>

## 6. Optional puzzle: the flat column

In section 4, lambda = 30 gives **exactly** 1.029 for every k from 1 to
16, and that's *worse* than section 2's biases-only model, which scores
0.969 on the same validation window. Explain both facts.

<details>
<summary>Check yourself</summary>

1. **Same number for every k.** The vectors start tiny (0.1 x random)
and the penalty on them is so strong that the ridge solves shrink them
to essentially zero, so p_u . q_i contributes nothing and the model
reduces to mu + b_u + b_i. That doesn't depend on k.

2. **Worse than biases only.** In this ALS the same lambda also
penalises the biases. Biases-only (section 2) uses a penalty of 5; at
30, a rider with 12 ratings keeps only 12 / 42 of their true
generosity, and a place with 40 ratings 40 / 70 of its quality. Too
much shrinkage throws away real signal. (Check: biases-only with a
penalty of 30 also scores 1.029 on validation.) It's a reminder that one
penalty for everything is a convenience, not a law: many systems use
separate penalties for biases and factors.

</details>
