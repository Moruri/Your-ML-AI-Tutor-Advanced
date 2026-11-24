# Lesson 016 - Collaborative filtering and matrix factorisation

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 3 | Tuesday 2026-11-24

> **Goal:** Factor a user-item matrix, predict missing ratings, and
> discuss cold start.

Time: about 75 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 015 ranked places with counts and a hand-made taste profile. It
never asked the most useful question in recommendation: **who else
liked what you liked?** If the riders who loved the same three parks as
you also loved a fourth, you'll probably like it too, and you don't
need to know it's a park. That idea, learning from the pattern of
everyone's choices rather than from descriptions of the items, is
**collaborative filtering**.

Today we use it to predict the star rating a rider would give a place
they haven't rated, in three steps of increasing ambition: averages and
biases, neighbours, and **matrix factorisation**, which squeezes every
rider and every place into a handful of numbers. Then we look at the
part every real system struggles with: the rider who signed up this
morning and the café that opened yesterday.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-016-collaborative-filtering-and-matrix-factorisation/lesson.py
```

It runs in about five seconds and saves `output/factors.png`: the
learned place factors coloured by genre on the left, the cold-start
curve on the right.

## 1. The rider-place matrix

Same city as lesson 015, a few months on. The app now asks for a star
rating after each visit, so riders rate over half of their visits (the
extremes still a bit more often). Two other changes: riders are busier,
and some of them are **generous** markers while others never give a
five, which is true of every rating system ever built.

```
  ratings / riders / places                      -> 18007 / 1200 / 240
  matrix filled in                               -> 6.3%
  ratings per rider: median / max                -> 12 / 77
  star counts 1..5                               -> 1003 2625 4432 5539 4408
  train / validation / test ratings              -> 9932 / 1109 / 2412
```

Picture a table with 1,200 rows (riders) and 240 columns (places),
93.7% empty. Our job is to fill in the blanks. Splits, as always, are
by time: train on days 0-269, pick settings on days 270-299, and test
on the last 65 days. We also lock 300 riders away completely for the
cold-start section. The score is **RMSE in stars**; lesson 015 told us
to distrust it as the final word, and we'll check ranking too.

## 2. Baselines: the mean, and biases

```
  global mean                                    -> 1.193
  raw place average                              -> 1.070
  mean + rider bias + place bias (shrunk)        -> 0.979
```

The third line is the workhorse of every recommender, and it's worth
writing down:

    predicted stars = mu + b_rider + b_place

`mu` is the overall average, `b_place` is how much better or worse than
average this place gets rated, and `b_rider` is how generous this rider
is. Both biases are **shrunk** (each is a ridge estimate, an average of
residuals divided by count + 5), lesson 015's fix for the raw average
applied to riders as well as places. Already a big step: almost all of
the easy signal in ratings is "this place is good" and "this rider is
generous". The interesting part, *which riders like which kind of
place*, is what's left.

## 3. Neighbourhood CF: places like the ones you rated

**Item-item collaborative filtering** fills a blank like this:

1. Remove the biases from every rating, so what's left is "how much
   more this rider liked this place than expected".
2. Measure how similar two places are by the cosine between their
   columns of leftovers, over the riders who rated both. Shrink the
   similarity towards zero when only a few riders rated both
   (multiply by n / (n + 10)), or two places with one shared fan look
   like twins.
3. To predict rider u at place i, take the k places most similar to i
   that u has rated, and average u's leftovers on them, weighted by
   similarity. Add the biases back.

```
  k =  5 neighbours: validation RMSE             -> 0.965
  k = 20 neighbours: validation RMSE             -> 0.957
  k = 50 neighbours: validation RMSE             -> 0.957
  k = 20, refit, test RMSE                       -> 0.944
  place 123 (nature): 5 nearest                  -> 81 (nature), 128 (nature), 188 (culture), 203 (nature), 8 (food)
```

A modest gain over biases (0.979 to 0.944), and a nice property: the
neighbours are mostly the same genre, though nobody told the algorithm
about genres. It's also easy to explain to a rider ("because you liked
Uhuru Gardens"), which is why item-item CF ran at Amazon for years.
Its weakness is the sparsity: two places only become neighbours if
enough riders rated both.

## 4. Matrix factorisation

Matrix factorisation goes one step further. Give every rider a short
vector `p_u` and every place a short vector `q_i`, both of length k,
and model

    predicted stars = mu + b_rider + b_place + p_u . q_i

The dot product is the "taste match": if a place's vector points in
the same direction as a rider's, the rating goes up. Nobody chooses
what the k directions mean; they're learned so that the predictions
fit the ratings we have. With 1,200 riders, 240 places and k = 2,
that's about 4,300 numbers (vectors and biases) to describe a table
of 288,000 cells.

How to fit it? If the place vectors were known, each rider's vector
(and bias) would be a small **ridge regression** on the places they
rated. And if the rider vectors were known, each place's would be too.
So **alternating least squares (ALS)** does exactly that: solve every
rider given the places, then every place given the riders, and repeat.
Each step can only lower the penalised error, and twelve rounds are
plenty here. The penalty `lambda` (on the vectors and biases) is what
stops a rider with three ratings from getting a wild vector.

```
  k =  1, lambda 1 / 3 / 10 / 30                 -> 0.910 / 0.905 / 0.957 / 1.029
  k =  2, lambda 1 / 3 / 10 / 30                 -> 0.926 / 0.901 / 0.953 / 1.029
  k =  4, lambda 1 / 3 / 10 / 30                 -> 0.986 / 0.904 / 0.948 / 1.029
  k =  8, lambda 1 / 3 / 10 / 30                 -> 1.034 / 0.929 / 0.948 / 1.029
  k = 16, lambda 1 / 3 / 10 / 30                 -> 0.993 / 0.921 / 0.948 / 1.029
  chosen k = 2, lambda = 3: test RMSE            -> 0.876
```

Read the grid by columns:

- **lambda = 1 overfits**, and more so as k grows: k = 8 is worse than
  the biases-only baseline. More factors means more freedom to memorise
  the few ratings each rider has.
- **lambda = 30 kills the factors.** Every k gives the same 1.029, because
  the penalty shrinks every vector to zero and leaves only (heavily
  shrunk) biases.
- In between, a **small k is enough**. The city has four genres and a
  distance effect, but with a median of 12 ratings per rider, the data
  can only pin down about two directions reliably.

The chosen model beats the neighbours clearly (0.944 to 0.876), and
the biases by more (0.979 to 0.876). That's the typical ordering.

There's a shortcut that looks like the same thing and isn't:

```
  trap: zero-filled SVD, k = 4: test RMSE        -> 2.666
    ... its mean prediction on test pairs        -> 0.44 stars (truth 3.45)
```

Filling the blanks with zeros and taking a truncated SVD (the textbook
low-rank approximation) treats "hasn't rated" as "zero stars", and 94%
of the cells are blanks. So it learns to predict almost nothing. The
difference that makes MF work is that its loss **only counts the cells
we observed**. (Lesson 017 comes back to blanks, because for clicks
"no click" *does* carry some information.)

## 5. What did the factors learn?

```
  R^2 of 'food' from factors, k = 2 / k = 8      -> 0.21 / 0.37
  R^2 of 'nature' from factors, k = 2 / k = 8    -> 0.12 / 0.45
  R^2 of 'culture' from factors, k = 2 / k = 8   -> 0.27 / 0.47
  R^2 of 'nightlife' from factors, k = 2 / k = 8 -> 0.20 / 0.46
  corr(place bias, true average enjoyment)       -> 0.86
```

We never gave the model genres, but we can ask how well a straight line
from a place's factors recovers its true genre loading. With k = 2,
partly: two directions have to share four genres and the neighbourhood
effect. With k = 8, about half of each genre's variation is in there.
That's worth pausing on. The k = 8 model has *better* factors and
*worse* predictions than k = 2: with this little data, a smaller model
that's right about the big things beats a bigger one that's noisy about
everything. The place biases, meanwhile, line up well with true
quality (0.86).

The left panel of the figure shows the place factors; the genres form
loose clouds rather than clean clusters, which is what "partly" looks
like.

And the ranking check, to keep lesson 015's promise. For each test
rider with at least three test ratings, we sort *the places they
actually rated* by predicted stars and score the order with NDCG@5,
using stars - 1 as a graded relevance:

```
  NDCG@5 within a rider's test places: random order -> 0.865
  NDCG@5 within a rider's test places: place bias only -> 0.935
  NDCG@5 within a rider's test places: item-item neighbours -> 0.946
  NDCG@5 within a rider's test places: matrix factorisation -> 0.957
    (riders with 3+ test ratings)                -> 376
```

Same order as RMSE. But note the question this answers: "among the
places you went to, which did you like most?" It is **not** "which of
the 240 places should we show you?". Lesson 015's trap 2 is still in
force: the ratings only cover places people chose, so a star model
can't tell you what people would do with the rest. That's lesson 017's
problem.

## 6. Cold start

Collaborative filtering learns from history, so it has nothing to say
about **a rider or a place with no history**. Every real system needs
a plan for both.

**New riders.** Keep the place vectors fixed, and as the new rider
rates places, solve the same small ridge regression ALS uses to get
their vector and bias (this is called **folding in**, and it takes a
millisecond, no retraining). We do it for the 300 locked-away riders,
using their first n ratings, and score them on their 11th rating
onwards:

```
  n =  0: fold-in MF / biases only               -> 1.044 / 1.044
  n =  1: fold-in MF / biases only               -> 1.011 / 1.018
  n =  2: fold-in MF / biases only               -> 0.987 / 1.006
  n =  3: fold-in MF / biases only               -> 0.958 / 0.991
  n =  5: fold-in MF / biases only               -> 0.927 / 0.977
  n = 10: fold-in MF / biases only               -> 0.870 / 0.956
    (new riders with 15+ ratings)                -> 125
```

With zero ratings, all you can do is predict each place's average,
which is why apps lean on **popularity** for new users, and why so
many onboarding screens ask you to "pick three things you like". Each
rating is worth something, and by ten the new rider is roughly as well
served as the established riders in section 4 (0.870 here against
0.876 there, on different riders, so call it a tie). The ridge penalty matters here: with
one rating, an unpenalised fit would give the rider a vector pointing
wherever that one place happens to point.

**New places.** A new place has no ratings at all, so neither biases
nor factors exist. But it does have **side information**: the owner
picked a genre and gave an address when signing up. Learn a linear map
from (genre, zone) to the place vectors and biases, using all the
places we know, and apply it to the new one (each place is left out of
its own map, as if it were new):

```
  rider bias only (place unknown)                -> 1.094
  genre + zone tags -> place factors             -> 1.021
  the same places once rated (full MF)           -> 0.876
```

Tags close about a third of the gap. The rest is **quality**, which no
signup form captures: you only find out by showing the place to riders
and reading the stars. That's lesson 015's trap 3 again. A system that
never shows new places never learns about them, so production
recommenders deliberately give new items some exposure.

## What to remember

- **Biases first.** mu + rider bias + place bias, shrunk, is the
  baseline every CF model must beat, and it gets much of the way.
- **Item-item CF** recommends places similar to those you liked, where
  "similar" comes from co-ratings. Shrink similarities built on few
  shared riders.
- **Matrix factorisation** describes every rider and place with k
  learned numbers and predicts with a dot product. Fit it with **ALS**:
  alternate small ridge regressions.
- Tune **k and lambda together**, on a time-based validation window.
  Too little penalty overfits; too much leaves only the biases.
- Only fit the **observed** cells. Zero-filled SVD treats missing as
  "hated it".
- **Cold start** needs its own plan: popularity and fold-in for new
  riders, side information and deliberate exposure for new places.
- Star ratings answer "how much will they like it if they go?", not
  "where will they go?".

## What to do now

1. Run `lesson.py` and look at `output/factors.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 017 drops the stars and learns from clicks and visits instead,
   adds content features, and trains with a ranking loss, so that the
   model is optimised for the list a rider actually sees.
