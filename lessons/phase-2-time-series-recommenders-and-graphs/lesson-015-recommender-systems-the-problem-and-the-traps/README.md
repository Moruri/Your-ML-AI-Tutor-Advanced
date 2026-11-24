# Lesson 015 - Recommender systems: the problem and the traps

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 3 | Tuesday 2026-11-24

> **Goal:** Frame recommendation as predict-and-rank, and measure it with
> ranking metrics instead of accuracy.

Time: about 60 minutes to read and run, plus the exercises. Nothing new
to install.

---

The bike-share app is getting a new tab. It's called **Explore**, and
it shows each rider ten places they might like to cycle to next: a
café, a park, a market, a gallery, a bar. The product team has a year
of logs (who rode to which place, and when), a few thousand star
ratings, and a simple question: which ten places do we show?

That sounds like classification ("will this rider go there, yes or
no?"), and the first trap is treating it like one. A rider sees ten
slots, not two hundred and forty yes/no answers. What matters is
**which places land at the top of the list**. So the job is: score
every place for every rider, sort, show the top k, and judge the
*list*. Today we build the scoreboard for that job, run a few honest
baselines, and walk into the three traps that make recommenders look
better in a notebook than they do in the app. Lesson 016 builds the
first real model.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-015-recommender-systems-the-problem-and-the-traps/lesson.py
```

It runs in a few seconds and saves `output/recs.png`: the long tail of
visits on the left, the baselines' scores on the right.

## 1. What the app has logged

The simulation builds 240 places and 1,200 riders. Each place has a main
genre (food, nature, culture, nightlife), a hidden quality, a
neighbourhood along the city's long main road, a **visibility** (good
signage, on the main road, all over Instagram) and, for 15% of them, an
opening day partway through the year. New places get a two-month buzz.
Each rider has a taste mix (most people love one or two genres), a home
neighbourhood (far places are less appealing) and an activity level.

From those, every rider has a hidden **enjoyment** score for every
place. Visits are drawn with probability proportional to enjoyment
times visibility: people go where they'd have fun, *and* where they've
heard of. A real app never sees the enjoyment, only the visits:

```
  riders / places / visits                       -> 1200 / 240 / 13036
  visits per rider: median / max                 -> 8 / 71
  share of rider-place pairs ever visited        -> 4.5%
  visits to the top 10% of places                -> 39%
  visits with a star rating                      -> 3351 (26%)
```

Two numbers set the scene for every recommender you'll meet. The matrix
is **sparse**: 95.5% of rider-place pairs are empty. And attention is
**concentrated**: a tenth of the places get two-fifths of the visits
(the left panel of the figure is the classic long tail).

## 2. Accuracy is the wrong scoreboard

We split by **time**, like every forecast in lessons 011-014: train on
days 0-299, and ask what each rider does in the last 65 days. Test
riders are those with some history and some future.

```
  test riders / future visits to predict         -> 889 / 2210
  accuracy of predicting 'no visit' for every pair -> 98.92%
  useful places it recommends                    -> 0
```

A model that says "nobody goes anywhere" is 98.9% accurate and
completely useless. With rare positives, accuracy rewards saying no,
and here the positives are very rare: a rider does a handful of things
out of hundreds.

## 3. Ranking metrics, by hand

So we score the list. For each rider, take the top k places they
haven't already been to, and compare with where they actually went
later. Then average over riders. The usual set:

| Metric | Question it answers |
|---|---|
| **precision@k** | What share of the k slots were places the rider went to? |
| **recall@k** | What share of the rider's future visits made the list? |
| **hit rate@k** | Did the list contain at least one right answer? |
| **NDCG@k** | Were the right answers near the *top*? |
| **coverage** | What share of all places appears in anyone's list? |

NDCG needs a small worked example. Each hit at position r earns
1 / log2(r + 1), so position 1 is worth 1, position 3 is worth 0.5, and
position 10 about 0.29. Add those up (that's the **DCG**) and divide by
the best possible DCG for this rider, where all their future visits sit
at the top. A list of 5 with hits at positions 1 and 3, for a rider who
made 3 future visits:

```
  precision@5 / recall@5                         -> 0.40 / 0.67
  DCG = 1/log2(2) + 1/log2(4)                    -> 1.500
  ideal DCG (3 hits at the top)                  -> 2.131
  NDCG@5                                         -> 0.704
  same 2 hits, at positions 3 and 5: NDCG@5      -> 0.416
```

Same precision, same recall, very different NDCG: it's the one metric
here that cares about order *within* the list, which is what a rider
scrolling the tab experiences. Coverage is the odd one out: it's not
about any one rider, it's about whether the system ever shows most of
the catalogue.

## 4. Baselines on an honest split

Five ways to fill the ten slots. None of them learns anything clever:

- **random**, to know what zero skill looks like;
- **most popular**: the same ten places for everyone (minus the ones
  you've been to);
- **popular in your zone**: counts among riders from your neighbourhood;
- **your genres x popularity**: the genre mix of your past visits,
  times zone popularity. A hand-made taste profile;
- **true enjoyment (oracle)**: the hidden scores. Not available in real
  life; here as a reference.

```
  random                                         -> P@10 0.012 | R@10 0.046 | NDCG@10 0.027 | hit 11% | cover 100%
  most popular                                   -> P@10 0.046 | R@10 0.201 | NDCG@10 0.134 | hit 38% | cover 13%
  popular in your zone                           -> P@10 0.058 | R@10 0.253 | NDCG@10 0.173 | hit 46% | cover 26%
  your genres x popularity                       -> P@10 0.066 | R@10 0.285 | NDCG@10 0.194 | hit 49% | cover 60%
  true enjoyment (oracle)                        -> P@10 0.083 | R@10 0.359 | NDCG@10 0.234 | hit 60% | cover 85%
  future visits that go to places opened since day 240 -> 18%
```

Read that slowly, because it's the shape of almost every recommender
result you'll see:

1. **Popularity is a strong baseline.** One list for everyone gets five
   times random's NDCG. Any model you build has to beat *this*, not
   random, or it isn't worth running.
2. **Small, obvious signals pay.** Location adds about 30%, a crude taste
   profile a bit more, and coverage jumps from 13% to 60% on the way.
3. **The numbers look small, and that's normal.** Precision@10 of 0.066
   means about two-thirds of a useful place per list. Riders only made
   2.5 visits each in the test window; most slots *can't* be hits.
4. **Even the oracle isn't perfect.** It knows exactly what each rider
   would enjoy, but visits also depend on visibility and on new-place
   buzz, which enjoyment ignores. Predicting *what people do* and
   *what people would like* are different targets. Keep that in mind;
   it comes back in lesson 017.

And 18% of the future visits go to places that opened in the last two
months of training, which a popularity count has barely seen. That's
the new-item problem in one line.

## 5. Trap 1: the random split

The most common mistake in recommender papers and notebooks: hold out a
random 18% of all visits (the same size as our test set) instead of the
last 65 days.

```
  most popular, time split                       -> NDCG@10 0.134 | hit 38%
  most popular, random split                     -> NDCG@10 0.217 | hit 51%
```

Same method, 60% better on paper. The random split lets the popularity
count include visits from November when it's "predicting" October, so
it already knows which new places took off. It's lesson 012's random
cross-validation leak in a new costume. **Split by time**, or at least
by "everything after this rider's cut-off day", and your offline number
will be closer to what the app sees.

## 6. Trap 2: "sort by average rating"

Stars feel like the obvious ground truth. They're not, because people
**choose what they rate**. They rate places they picked (so they liked
them more than average), and they rate the extremes more often than
the middle. The simulation knows what every rider *would* give every
place; the app only sees the stars people chose to leave:

```
  mean star given / share of 4-5 stars           -> 4.13 / 79%
  if every rider rated every place (hidden)      -> 2.64 / 23%
```

Ratings are **missing not at random**. The observed average says the
city is wonderful; the full matrix says most places would bore most
riders. Now try the classic "top rated" list. We judge each list by
the places' true rank on average enjoyment (1 = best of 240):

```
  places with 1-3 ratings                        -> 53
  top 10 by raw average                          -> mean true rank  72.6 of 240, median ratings each    7
  top 10 by shrunk average (m=10)                -> mean true rank   6.9 of 240, median ratings each   60
  top 10 by visit count                          -> mean true rank  18.4 of 240, median ratings each   62
```

The raw average crowns places with a handful of five-star ratings,
which is luck, not quality. The fix is lesson 005's partial pooling in
one line: pretend every place starts with m = 10 ratings at the city
average,

    shrunk = (sum of stars + m * city mean) / (number of ratings + m)

so a place needs plenty of good ratings to climb. The shrunk list's
average true rank is 6.9; the raw one's is 72.6.

## 7. Trap 3: the feedback loop

Offline evaluation assumes the logs don't depend on the recommender.
Once the tab is live, they do: riders mostly visit what they're
**shown**, and next week's "most popular" is computed from those
visits. We simulate twelve weeks. Each week every rider sees ten places
and sometimes rides to one:

- **A**: the ten most-visited places in your zone, from the app's own logs;
- **B**: the top seven, plus three random picks from the zone's top 60.

```
  A: top 10 by your zone's visits                -> places ever shown 12%, app visits to top 10% 96%, enjoyment wk 1-2 1.71 -> wk 11-12 1.73
  B: top 7 + 3 random picks                      -> places ever shown 61%, app visits to top 10% 89%, enjoyment wk 1-2 1.66 -> wk 11-12 1.68
```

Policy A is a closed loop. It only ever shows 12% of the places, so
96% of its visits go to the top tenth, so those places stay on top.
The other 88% of the catalogue never gets a chance, and **the logs
can't tell you whether they're any good**, because nobody was shown
them. Policy B gives up a little enjoyment (1.68 against 1.73) to show
five times as much of the city, and its logs now contain evidence about
places A will never learn about. That trade between using what you
know and finding out more is called **exploration versus exploitation**;
it's the heart of the reinforcement-learning lessons in phase 3.

## What to remember

- A recommender **scores and ranks**. Judge the top of the list, not
  every pair: accuracy rewards recommending nothing.
- Know your metrics: **precision, recall and hit rate** count hits;
  **NDCG** rewards putting them first; **coverage** asks whether the
  catalogue gets seen at all.
- **Popularity is the baseline to beat.** Location and a crude taste
  profile already beat it here.
- **Split by time.** A random split leaks the future and flattered
  popularity by 60%.
- **Ratings are missing not at random.** Never sort by a raw average;
  shrink it towards the overall mean.
- **Live recommenders shape their own data.** Without some exploration,
  the logs only confirm what you already showed.

## What to do now

1. Run `lesson.py` and look at `output/recs.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 016 builds the first learned recommender: collaborative
   filtering and matrix factorisation on the star ratings, and what to
   do when a rider or place has no history at all.
