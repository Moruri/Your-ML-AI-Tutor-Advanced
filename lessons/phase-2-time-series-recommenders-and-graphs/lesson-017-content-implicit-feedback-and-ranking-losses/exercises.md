# Lesson 017 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_017
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-017-content-implicit-feedback-and-ranking-losses/lesson.py my_work_017/
cd my_work_017
python lesson.py
```

---

## 1. Confidence by hand

The lesson uses confidence c = 1 + alpha x log(1 + rides), with
alpha = 3.

a) What confidence do cells with 0, 1, 4 and 39 rides get?
b) With the *linear* version, c = 1 + alpha x rides, what do the 4-ride
   and 39-ride cells get?
c) In words: what does the log protect against?

<details>
<summary>Check yourself</summary>

a) 0 rides: 1 + 3 x log(1) = **1**. 1 ride: 1 + 3 x 0.693 = **3.08**.
4 rides: 1 + 3 x 1.609 = **5.83**. 39 rides: 1 + 3 x 3.689 = **12.1**.

b) 1 + 3 x 4 = **13** and 1 + 3 x 39 = **118**.

c) One obsessive rider. Under the linear rule the person who rode to
the same bar 39 times gets almost ten times the pull of someone who
went 4 times, so a few heavy repeaters can bend the place's vector
towards their own taste. The log says "going back is evidence, but the
tenth visit tells you less than the second". Whether that's right is
an empirical question (see exercise 4).

</details>

## 2. One BPR step

For rider u, visited place i scores 1.2 and unvisited place j scores
0.4.

a) What is this triple's loss, -log sigmoid(s_i - s_j)?
b) The gradient of log sigmoid(gap) with respect to the gap is
   1 - sigmoid(gap). What is it here, and what is it when the gap is
   4.0, or -2.0?
c) What does that tell you about which triples BPR learns from?

<details>
<summary>Check yourself</summary>

a) Gap = 0.8. sigmoid(0.8) = 0.690, so the loss is -log 0.690 =
**0.371**.

b) Gap 0.8: 1 - 0.690 = **0.31**. Gap 4.0: 1 - 0.982 = **0.018**.
Gap -2.0: 1 - 0.119 = **0.88**.

c) Triples already in the right order by a wide margin contribute
almost nothing; mis-ordered ones (visited place below the unvisited one)
get nearly full weight. BPR spends its effort where the list is wrong.
That's also why it gets slower late in training: most random triples
are easy (a visited favourite versus a random far-away place), so
smarter negative sampling is an active research area.

</details>

## 3. The random baseline

Random scores 0.041 recall@10.

a) Test riders have visited about 13 distinct places by day 300. Estimate
   random's recall@10 from that alone.
b) Why do we remove places a rider has already visited from the
   candidates? Name a product where you would *not* remove them.

<details>
<summary>Check yourself</summary>

a) A random list of 10 from about 240 - 13 = 227 candidates contains
each future discovery with probability 10 / 227 = **0.044**, and
recall@10 is the average of that over a rider's discoveries. Close to
0.041 (riders vary in how many places they've seen).

b) The lesson's task is *discovery* (where will they go next that's
new?) and the Explore tab wants new places. Re-recommending last
week's café would score well on "will they visit?" and be useless to
the rider. Where repeats are the point (groceries, a music playlist,
reordering a regular meal) you'd keep them, and often a "buy it again"
baseline is extremely hard to beat.

</details>

## 4. Hands-on: how much should a repeat visit count?

In your copy, change the confidence in `implicit_als` from
`1 + alpha * np.log1p(C)` to the linear `1 + alpha * C`. Keep the chosen
k = 32 and lambda = 30, and run the test evaluation with alpha = 1 and
alpha = 3.

Predict first: will the linear rule be better or worse than the log?

<details>
<summary>Check yourself</summary>

What you should see (test NDCG@10; the log version scored 0.172):

    linear, alpha = 1   -> R@10 0.245 | NDCG@10 0.163
    linear, alpha = 3   -> R@10 0.269 | NDCG@10 0.179

Linear with alpha = 3 is a little **better** here. In this simulated
city, repeat visits are driven by enjoyment and nothing else, so a
heavy repeater is a clean signal and there's nothing to protect
against. In a real app, counts are messier (a commuter's daily ride to
the station they work near, an app that auto-plays the next song), and
the log or a cap often wins. The general lesson: the confidence
function is a hyperparameter. Choose it on the validation window like
k and lambda, never on test as we've just done.

</details>

## 5. Hands-on: negatives from the popular end

BPR samples its unvisited place j uniformly. Many real systems sample
negatives **in proportion to popularity** (to the power 0.75, as in
word2vec), on the grounds that "a popular place you skipped" is a more
informative negative than an obscure one.

In your copy of `train_bpr`, replace

```python
j = r.integers(0, I, len(idx))
```

with

```python
j = r.choice(I, len(idx), p=pw_)
```

where, before the loop, `pop = (C > 0).sum(0) + 1.0` and
`pw_ = pop ** 0.75 / (pop ** 0.75).sum()`. Retrain plain BPR and
evaluate on test.

<details>
<summary>Check yourself</summary>

    BPR, popularity-sampled negatives -> R@10 0.167 | NDCG@10 0.109

From 0.164 down to 0.109, about the level of the popularity baseline.
Popular places now appear as negatives all the time, so the model
learns to push them **down**, but in this city (and our test) popular
places really are where people go next. You've made a model that
fights the strongest signal in the data.

That's not to say the idea is wrong: popularity-aware negatives can
help when you want to reduce popularity bias in the list or when the
test measures something other than raw next-visits. It's a choice with
consequences, and the scoreboard decides.

</details>

## 6. Optional puzzle: how long to train?

The lesson trains BPR for 40 epochs. Retrain with 5 epochs and with 80
and evaluate both on test. Then explain: why does a model trained on
*random* triples need so many passes, and why might more passes make it
slightly worse?

<details>
<summary>Check yourself</summary>

    5 epochs   -> R@10 0.170 | NDCG@10 0.107
    40 epochs  -> R@10 0.236 | NDCG@10 0.164   (the lesson)
    80 epochs  -> R@10 0.225 | NDCG@10 0.154

1. **Slow start.** Each epoch shows every visited pair once, against
   one random unvisited place. Most random negatives are easy (exercise
   2), so each step moves the vectors only a little. After 5 epochs the
   model has learned popularity (the bias b_i is the fastest thing to
   learn) and little else: 0.107, the popularity baseline.
2. **Slight decline.** By 80 epochs it has started to fit the
   particular places each rider visited before day 300, which the small
   regularisation (0.01) doesn't fully stop. That's overfitting in a
   ranking model.

The fix is the one from every lesson since phase 1: pick the number of
epochs (or use early stopping) on the **validation** window, and
report test once.

</details>
