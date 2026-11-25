# Lesson 017 - Content, implicit feedback and ranking losses

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 4 | Wednesday 2026-11-25

> **Goal:** Build a recommender from clicks rather than stars, and
> train it with a ranking objective.

Time: about 75 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 016 ended with an awkward truth: star ratings tell you how much
someone liked a place *once they went*, and most riders never rate
anything. What every app has in abundance is behaviour. Rider 412 rode
to the Kariakor market on Saturday, and again the Saturday after. Rider
77 opened the Explore tab, saw a gallery, and didn't go. That stream of
clicks, rides and repeat visits is called **implicit feedback**, and
it's what most real recommenders run on.

It's a different kind of data, and it breaks lesson 016's recipe in an
instructive way. A visit is a strong hint that the rider likes a place;
a *missing* visit is a weak, muddled hint (maybe they'd hate it, maybe
they've never heard of it). Today we learn from that with two classic
models, **weighted implicit ALS** and **BPR**, a loss that cares only
about the order of the list. Then we bolt on content (each place's
genre and neighbourhood) so a café that opened last week isn't
invisible.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-017-content-implicit-feedback-and-ranking-losses/lesson.py
```

It runs in under ten seconds and saves `output/ranking.png`: every
model's test score next to the oracle on the left, and one rider's
recommendations against the popularity curve on the right.

## 1. Clicks, not stars

Same city as lessons 015 and 016: 240 places, 1,200 riders, each rider
with a hidden enjoyment score for every place. This time the log has
**no ratings at all**, just rides, and riders go back to places they
like. Thirty places open late in the year, right around the end of the
training window.

```
  riders / places / rides                        -> 1200 / 240 / 25989
  distinct rider-place pairs                     -> 17302 (6.0% of the matrix)
  rides that were repeat visits                  -> 33%
  rides per pair: median / max                   -> 1 / 39
  mean rides when enjoyment is top / bottom quarter -> 1.63 / 1.01
  never-visited pairs in the riders' top quarter -> 81%
  validation / test riders with a new place to find -> 736 / 1005
```

Two lines carry the lesson. Repeat visits **mean something**: pairs
the rider truly enjoys get visited 1.63 times on average, the weak
ones barely more than once. And blanks **don't mean dislike**: 81% of
the places each rider would enjoy most have never been visited. A blank
is mostly "hasn't got round to it yet". That's the gap we're trying to
fill.

The task: for each rider, rank the places they *haven't* visited yet,
and see whether the ones they discover next land in the top 10. Time
split as always (train days 0-269, tune on 270-299, refit on 0-299,
test on 300-364), scored with lesson 015's recall@10 and NDCG@10.

## 2. Baselines, and the trap of fitting only what you saw

```
  random                                         -> R@10 0.041 | NDCG@10 0.023
  most popular                                   -> R@10 0.168 | NDCG@10 0.110
  MF on observed cells only (lesson 016 style)   -> R@10 0.167 | NDCG@10 0.108
    spread of its scores (std / mean)            -> 0.278 / 0.786
```

Popularity is, again, a stubborn baseline. The interesting line is the
third. Lesson 016's golden rule was "only fit the observed cells". Here
every observed cell is a **1** (they went). A model trained to predict
1 everywhere it looks has nothing to learn about taste: any vectors
whose dot products come out near 1 will do. What it ends up with is
roughly "places lots of riders went to point in the common direction",
which is popularity again, scored 0.108 against popularity's 0.110.
With implicit data you **must** use the blanks; the question is how
much to trust them.

## 3. Weighted implicit ALS

Hu, Koren and Volinsky's answer (2008) is beautifully simple. Turn
every cell of the matrix into two numbers:

- a **preference** p = 1 if the rider ever went, else 0;
- a **confidence** c = 1 + alpha x log(1 + rides).

Then fit `p ≈ x_u . y_i` over **all** 288,000 cells, weighting each
squared error by its confidence. A blank still counts (it says "probably
0"), just with the lowest weight; a place someone visited ten times
counts a lot. It's still ALS: each rider's vector is a weighted ridge
regression on all 240 places. The trick that keeps it fast is that the
all-blank part, `Y^T Y`, is the same for every rider, so each solve only
adds a correction for the handful of places that rider visited.

```
  k =  8, (lam, alpha) = (3,3) (3,15) (30,3) (30,15) -> 0.178 / 0.171 / 0.206 / 0.199
  k = 16, (lam, alpha) = (3,3) (3,15) (30,3) (30,15) -> 0.149 / 0.143 / 0.206 / 0.178
  k = 32, (lam, alpha) = (3,3) (3,15) (30,3) (30,15) -> 0.115 / 0.092 / 0.207 / 0.160
  chosen k = 32, lam = 30, alpha = 3: test       -> R@10 0.260 | NDCG@10 0.172
  same, alpha = 0 (a visit weighs no more than a blank) -> R@10 0.169 | NDCG@10 0.112
```

Read the grid as in lesson 016: penalty and size have to be tuned
together. With a light penalty, bigger k gets **worse** fast (0.178 to
0.115), because it memorises who went where. With lambda = 30, k barely
matters (0.206-0.207). The validation scores are basically tied across
k at that penalty, so a smaller k would have been an equally good and
cheaper choice. On test, implicit ALS lifts NDCG from 0.110 to 0.172,
and recall@10 from 17% to 26%.

The last line is the opposite trap from section 2. With alpha = 0, a
visit weighs exactly as much as a blank, so the 94% of cells that are 0
drown the 6% that are 1, and we're back at popularity. **Fit only the
observed cells and you learn nothing; trust the blanks fully and you
learn nothing either.** The confidence weight is what sits between.

## 4. A ranking loss: BPR

ALS still squares errors on 0/1 numbers, a regression loss. But the
app never shows a predicted number. It shows a **list**. Rendle et
al.'s **Bayesian Personalised Ranking** (2009) trains for the list
directly. Pick a rider u, a place i they visited and a random place j
they didn't, and maximise

    log sigmoid( score(u, i) - score(u, j) )

"rank the visited place above the unvisited one". Nothing asks the
score to equal 1 or 0; only the **gap** matters. Train with stochastic
gradient steps on fresh random triples, epoch after epoch. Here
`score(u, i) = p_u . q_i + b_i`, the same factor model as lesson 016,
plus a place bias.

```
  BPR (pairwise ranking loss)                    -> R@10 0.236 | NDCG@10 0.164
  same model, pointwise logistic loss            -> R@10 0.229 | NDCG@10 0.152
  BPR + genre/zone tags (hybrid)                 -> R@10 0.235 | NDCG@10 0.164
  hidden truth: enjoyment x visibility (oracle)  -> R@10 0.442 | NDCG@10 0.322
```

Three honest observations.

- BPR beats the identical model trained **pointwise** (visited = 1,
  sampled blank = 0, logistic loss) on the very same triples, 0.164 to
  0.152. That's the ranking objective earning its keep, but it's a
  modest gain, not a revolution.
- Implicit ALS (0.172) edges out BPR here. That's common: well-tuned
  ALS is a very strong baseline, and SGD models like BPR are touchier
  (learning rate, epochs, how negatives are sampled; exercise 4 shows
  how much the last one matters). Always run both.
- Even the **oracle**, which knows every rider's true enjoyment and
  every place's visibility, only gets 0.322. Rides are random; a rider
  won't visit every place they'd love in a nine-week window. Your
  ceiling is far below 1, so compare models to each other and to the
  oracle, not to perfection.

The hybrid's tags (one-hot genre and zone) don't help **warm** places:
for a place with hundreds of rides, its learned vector already knows
everything the tags say. Tags matter where vectors can't be learned.

## 5. New places: content to the rescue

In the hybrid, a place's vector is `q_i + tags_i @ E`: a free part
learned from rides, plus a part built from its genre and zone through
a shared matrix `E` that every place helps to train. That's the
**LightFM** idea. For a place with no rides, the free part is unknown,
so we set it to zero and score from the tags alone. Here we rank only
the 20 places that opened too late to appear in training:

```
  places with no rides before day 300            -> 20
  share of test discoveries that are new places  -> 13%
  random order                                   -> R@5 0.218 | NDCG@5 0.135 | riders 288
  BPR (new places only ever seen as negatives)   -> R@5 0.254 | NDCG@5 0.155 | riders 288
  hybrid, content only for new places            -> R@5 0.569 | NDCG@5 0.389 | riders 288
```

One discovery in eight in the test window is a brand-new place. Pure
BPR has nothing to go on (its vectors for them never saw a positive
example), so it's barely better than shuffling. The hybrid, knowing
only "this is a nightlife place in zone 3", more than doubles recall.
That's the lesson 016 cold-start story in its implicit form: tags carry
**taste match**, and only exposure will reveal **quality**.

## What to remember

- **Implicit feedback** (rides, clicks, plays, purchases) is plentiful
  and biased: a visit is a strong positive, a blank is a weak, mixed
  signal, mostly "not yet".
- Don't fit only the observed cells (all ones) and don't treat blanks
  as firm zeros. **Weight** them: implicit ALS uses confidence 1 +
  alpha x f(count) on every cell.
- **Ranking losses** like BPR optimise the order of a pair, not a
  number. They match what the app shows, and help a little over the
  same model trained pointwise.
- Strong, well-tuned ALS is hard to beat; compare models on the **same
  time split** and against an oracle ceiling, not against 1.0.
- How you **sample negatives** is a modelling choice, with large
  effects.
- **Content features** (genre, zone, text, price) give new items a
  vector on day one. They don't help items with plenty of data, and
  they can't reveal quality.

## What to do now

1. Run `lesson.py` and look at `output/ranking.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 018 leaves riders and places for the bike network itself:
   stations joined by streets. We turn a network into an adjacency
   matrix and a feature matrix, and compute degrees, paths and
   centralities by hand, the groundwork for graph models in lesson 019.
