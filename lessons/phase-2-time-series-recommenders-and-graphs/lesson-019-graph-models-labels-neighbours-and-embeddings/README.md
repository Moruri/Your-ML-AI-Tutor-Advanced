# Lesson 019 - Graph models: labels, neighbours and embeddings

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 5 | Thursday 2026-11-26

> **Goal:** Predict a node label from its neighbourhood and learn a
> small graph embedding.

Time: about 75 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 018 ended with a hint: a station's neighbours look a lot like
the station. Today we use that hint to make predictions.

The city has grown to 600 docking stations. The operator wants to know
what kind of place each one serves: an **office** district, a
**residential** street, or a **leisure** spot by the park or the
water. It matters because it decides when the vans move bikes around.
Offices empty in the evening, leisure stations fill up on Saturdays. A
survey covered only 60 stations. The other 540 have some noisy hints
(share of rides in the morning peak, share at weekends, number of
docks, distance from the centre of town) and one thing that's easy to
get: the street map.

We go from the simplest possible idea to a small graph neural network,
writing each one by hand: a vote among neighbours, label propagation,
graph convolution as feature smoothing, a two-layer GCN with gradients
you can check, and node embeddings learned from random walks. Then we
use the same tools to guess **missing streets**, and finish with the
case where it all goes wrong.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-019-graph-models-labels-neighbours-and-embeddings/lesson.py
```

It takes about ten seconds and saves `output/graph_models.png`: the
stations by true type with the surveyed ones circled, the walk
embedding squashed to two dimensions, and the accuracy of every method.

## 1. The problem

```
  stations / streets / mean degree                   -> 600 / 2254 / 7.5
  connected components (largest)                     -> 6 (514)
  class counts (office, residential, leisure)        -> 160 / 217 / 223
  edge homophily: streets joining the same type      -> 76.9%  (chance would be 34.0%)
  surveyed (labelled) stations per split / splits    -> 60 / 20
```

This is **semi-supervised node classification**: one graph, a few
labelled nodes, and we want labels for the rest. It's also
**transductive**: the stations we predict are already in the graph
while we train, which is different from the usual "train on these
rows, predict on new rows". Keep that in mind; it comes back in
exercise 4.

The number that makes graph methods worth trying is **edge
homophily**: the share of edges joining two nodes of the same class.
Here it's 77%, against 34% if streets ignored type. Neighbourhoods are
made of similar places.

One survey of 60 is a noisy thing to judge a method on, so every
accuracy below is the mean over **20 random surveys**, measured on the
540 stations not surveyed.

## 2. Baselines, then a vote among neighbours

```
  majority                                           -> 34.6%  (+/- 3.8%)
  features only                                      -> 66.8%  (+/- 2.9%)
  1-hop vote                                         -> 55.8%  (+/- 3.5%)
  1-hop coverage                                     -> 49.9%  (+/- 2.0%)
  label propagation                                  -> 74.1%  (+/- 3.5%)
  features + LP scores                               -> 78.7%  (+/- 3.2%)
```

- **Features only** is ordinary logistic regression on the four
  hints, ignoring the map: 66.8%. Every graph method has to beat this,
  or it isn't earning its keep.
- **1-hop vote**: give each station the most common label among its
  surveyed neighbours. It looks like it should work with 77% homophily,
  but **only half** the stations have a surveyed neighbour at all. The
  rest fall back to "most common class", and the total is worse than
  ignoring the graph. Sparse labels are the normal case, not the
  exception.
- **Label propagation** fixes the reach problem. Start with the 60
  known labels as one-hot rows of a matrix Y0, then repeat

  ```
  F <- alpha * S @ F + (1 - alpha) * Y0
  ```

  where S = D^-1/2 A D^-1/2 is the adjacency matrix scaled by degree.
  Each step spreads label mass one hop further; the `(1 - alpha) * Y0`
  term keeps pulling the surveyed stations back to what we know. It
  converges to a closed form, F = (1 - alpha)(I - alpha S)^-1 Y0, and
  the script checks the two agree (max gap 1.4e-12). That's 74.1%,
  with no features at all.
- **Features + LP scores**: hand logistic regression the four features
  *and* the propagated label shares. 78.7%. The map and the hints
  carry different information, and using both beats either.

Label propagation has one blind spot worth seeing in the figure: the
graph has six components, and a small component with no surveyed
station receives no label mass at all. Graph methods can only move
information along edges that exist.

## 3. Graph convolution: smooth the features, then classify

The other way to use the graph is to change the **features**, not the
labels. Replace each station's features with an average over itself
and its neighbours:

```
Ahat = D^-1/2 (A + I) D^-1/2        (A + I: include yourself)
Z    = Ahat @ X                      (one hop of smoothing)
```

Then fit logistic regression on Z. Do it k times and each station's
features become an average over its k-hop neighbourhood. This is
**SGC**, simplified graph convolution (Wu et al., 2019), and it's a
graph neural network with the non-linearities taken out.

```
  logistic regression on Ahat^0 X                    -> 66.8%
  logistic regression on Ahat^1 X                    -> 78.1%
  logistic regression on Ahat^2 X                    -> 79.9%
  logistic regression on Ahat^3 X                    -> 79.5%
  logistic regression on Ahat^6 X                    -> 78.2%
  logistic regression on Ahat^20 X                   -> 75.0%
  spread of a feature across stations: hops 0 vs 20  -> 1.00 vs 0.32  (over-smoothing)
```

Why does averaging help? Each station's morning-peak share is a noisy
measurement of its type. Its neighbours are mostly the same type, so
averaging over eight of them cancels a lot of noise. It's the same
reason we averaged over 20 surveys.

Too much averaging hurts. After 20 hops the features are averages over
most of the city, the spread between stations has shrunk to a third,
and the classes blur together. This is **over-smoothing**, and it's
why most graph networks stop at two or three layers.

A **GCN** (Kipf and Welling, 2017) puts learned weights and a
non-linearity between the smoothing steps:

```
H   = relu(Ahat @ X @ W1)            (16 hidden units)
out = softmax(Ahat @ H @ W2)
```

The script trains one with full-batch Adam, dropout and weight decay,
with every gradient written out by hand. The loss only counts the 60
surveyed stations, but the forward pass runs over the whole graph,
which is how the unlabelled stations still shape what the labelled
ones see. Always check hand-written gradients:

```
  GCN gradient check: hand vs finite difference      -> -0.026009 vs -0.026009
  GCN training loss: epoch 1 / 300                   -> 1.296 / 0.298
  GCN accuracy on unlabelled                         -> 79.2%  (+/- 1.8%)
```

79.2%, the same as SGC within the noise. With four features and 60
labels there's not much for the extra layer to discover, and that's an
honest result you will meet often: on small graphs, the smoothing does
most of the work and the network adds little. The GCN is a bit more
stable across surveys (+/- 1.8%), thanks to dropout and weight decay.

## 4. Node embeddings from random walks

So far the graph helped a specific task. An **embedding** goes the
other way: learn a short vector for every node from the graph alone,
then reuse it for anything.

**DeepWalk** (Perozzi et al., 2014) treats random walks as sentences
and nodes as words. Nodes that turn up near each other on walks get
similar vectors, exactly like words in similar contexts. We skip the
neural network and do what it secretly does (Levy and Goldberg, 2014;
Qiu et al., 2018):

1. run 10 walks of 40 steps from every station;
2. count how often each pair of stations appears within 5 steps of
   each other;
3. turn the counts into **PPMI** (positive pointwise mutual
   information: log of "how much more often than chance", clipped at
   zero);
4. take the top 16 singular vectors.

```
  walks: count x length                              -> 6000 x 40
  share of walk steps that stay within a type        -> 79.0%
  embedding shapes (walk / spectral)                 -> (600, 16) / (600, 16)
  mean cosine: same-type pairs vs different          -> 0.10 vs 0.03
  walk embedding                                     -> 73.0%
  spectral embedding                                 -> 67.8%
  walk embedding + features                          -> 79.2%
```

The walk embedding never saw a label, yet logistic regression on it
alone reaches 73.0%, close to label propagation. Add the features and
it's 79.2%, level with the GCN. The **spectral embedding** (the
smallest non-trivial eigenvectors of the normalised Laplacian, a
classic) does worse here: it's dominated by the coarse structure,
which neighbourhood blob you're in, and the walks capture finer local
detail.

In the figure, the main component's embedding forms arms, roughly one
per neighbourhood. That's what an embedding of this kind learns: *where
you are in the graph*. It's not a summary of what you're like.

## 5. Link prediction: which streets are missing?

The same tools answer a different question: given the graph, which
**edges** should exist but don't? Recommenders (lesson 016) are link
prediction on a rider-place graph. Here we hide 10% of the streets and
see if we can find them again among an equal number of non-streets.

```
  AUC: straight-line distance (cheating-free baseline) -> 0.999
  AUC: common neighbours (trained without test edges) -> 0.965
  AUC: Adamic-Adar (trained without test edges)      -> 0.966
  AUC: walk-embedding cosine (trained without test edges) -> 0.944
  AUC: common neighbours, LEAKY full graph           -> 0.972
  AUC: walk-embedding cosine, LEAKY full graph       -> 0.964
```

- **Common neighbours**: two stations with many shared neighbours are
  probably joined. **Adamic-Adar** does the same but counts a shared
  neighbour less if it's a hub (everyone shares hubs).
- **Leakage**: compute the embedding on the full graph, hidden streets
  included, and the AUC goes up from 0.944 to 0.964. The model has seen
  the answers. Every link prediction evaluation must remove the test
  edges *before* computing any feature or embedding. It's the graph
  version of lesson 012's honest holdout.

Now the uncomfortable part. Straight-line distance gets 0.999 and
beats everything. Of course it does: in this city, streets *are* short
connections, and the random non-streets have a median distance of 4.6
against a cutoff of 0.42. Random negatives are far too easy. Pick
**hard negatives** instead, pairs that are two hops apart but not
joined, which is what a real "suggest a new street" tool has to sort:

```
  AUC with hard (2-hop) negatives: distance          -> 0.979
  AUC with hard (2-hop) negatives: common neighbours -> 0.728
  AUC with hard (2-hop) negatives: Adamic-Adar       -> 0.787
  AUC with hard (2-hop) negatives: walk-embedding cosine -> 0.694
```

Everything drops, and the ranking of methods becomes informative. Two
lessons: how you sample negatives decides what your AUC means, and a
plain domain feature (distance) can beat every graph trick. Always try
it.

## 6. When neighbours are unlike you

Homophily is an assumption, not a law. Build a different graph on the
same stations: join two stations if *lots of rides go between them*.
Commuters ride from home to work, so this **trip graph** mostly joins
residential stations to offices.

```
  trip graph: edges / homophily                      -> 1521 / 5.1%
  features only                                      -> 66.8%
  label propagation                                  -> 27.5%
  SGC (Ahat X, mixed)                                -> 50.2%
  own + neighbour-mean (kept apart)                  -> 71.5%
  own + neighbours' label shares                     -> 72.6%
```

Label propagation now does worse than guessing (27.5%), because it
confidently spreads each label to the *wrong* class. SGC averages
offices' features into residential stations and destroys the signal
(50.2%). Both bake in "you are like your neighbours".

The fix is simple: **keep your own features and your neighbours'
features in separate columns**, and let the model learn the
relationship. Then "my neighbours look like offices" can mean "I'm
residential". Accuracy goes *above* features-only (71.5%), and adding
the neighbours' known label shares gives 72.6%. This is the idea
behind heterophily-aware architectures (GraphSAGE's concatenation,
H2GCN), and it's cheap to test: measure homophily before you choose a
method.

## What to remember

- **Node classification with few labels** is the typical graph task.
  Always compare against **features only**.
- **1-hop votes** fail when labels are sparse. **Label propagation**
  spreads labels across the whole graph and has a closed form.
- **Graph convolution** = smoothing features over neighbours, then a
  normal model. Two hops is usually enough; more causes
  **over-smoothing**. A **GCN** adds weights and non-linearities and,
  on small graphs, often adds little.
- **Random-walk embeddings** (DeepWalk) are a factorised co-occurrence
  matrix: they encode *where* a node sits in the graph.
- **Link prediction**: remove test edges before computing anything,
  and use **hard negatives**, or the AUC is flattering nonsense.
- Check **homophily** first. Under **heterophily**, keep own and
  neighbour features apart.

## What to do now

1. Run `lesson.py` and look at `output/graph_models.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 020 is the phase 2 mini-project: pick one of forecasting,
   recommending or connecting, and take it all the way to a short,
   evaluated write-up.
