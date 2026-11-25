# Lesson 018 - Graphs as data

**Phase 2 - Time series, recommenders and graphs** | Week 2, Day 4 | Wednesday 2026-11-25

> **Goal:** Represent a network as an adjacency and a feature matrix,
> and compute degrees, paths and simple centralities.

Time: about 60 minutes to read and run, plus the exercises. Nothing new
to install (we use `scipy.sparse`, which you already have).

---

Every dataset so far has been a table: rows that don't know about each
other. Riders, places, days. But plenty of data is about the
**connections** between things. Stations joined by streets, people who
call each other, papers that cite papers, accounts that send each other
money. In that kind of data, *who you're connected to* is often the
most predictive thing about you, and a table quietly throws it away.

Today is the groundwork: how to hold a network in a computer, and the
handful of measurements everyone uses to describe one. We do it on the
bike-share network itself, 84 docking stations on two banks of a river
with only three bridges, plus a new suburb that isn't connected yet.
Each algorithm is written by hand in a few lines and then checked
against `scipy`, so nothing is a black box when lesson 019 starts
learning on graphs.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-2-time-series-recommenders-and-graphs/lesson-018-graphs-as-data/lesson.py
```

It runs in a couple of seconds and saves `output/network.png`: the map
of stations (sized by betweenness, coloured by degree) on the left, and
betweenness against degree on the right.

## 1. Two matrices

A **graph** is a set of **nodes** (stations) and **edges** (streets
between pairs of them). Edges can be **undirected** (a street works
both ways) or directed (a one-way street, a follow on social media),
and **weighted** (minutes to ride it) or not. Our streets are
undirected and carry a travel time.

Data usually arrives as an **edge list**, one row per edge. Two matrices
turn it into something linear algebra can use:

- the **adjacency matrix** A, n x n, with A[i, j] = 1 if i and j are
  joined (or the weight, if you want one);
- the **feature matrix** X, n x d, one row of ordinary features per
  node, exactly like the tables we're used to.

```
  stations / streets (undirected edges)          -> 84 / 193
  first three edges (station, station, minutes)  -> (0,2,3.7), (0,7,5.3), (0,8,2.4)
  A is symmetric / zero diagonal                 -> True / True
  density: edges / possible edges                -> 5.5%
  numbers stored: dense A / sparse A (CSR)       -> 7056 / 857
  1M stations, ~6 streets each: dense vs sparse  -> 8 TB vs about 104 MB
  feature matrix X: shape / columns              -> (84, 4) / docks, height_m, commute_share, near_rail
```

Undirected means **symmetric**: A[i, j] = A[j, i]. Real networks are
**sparse**: each station touches a handful of others, so 94.5% of A is
zeros. For 84 nodes that doesn't matter. For a million it's the
difference between 8 terabytes and about 100 megabytes, which is why
every serious graph library stores A in a sparse format such as
**CSR** (compressed sparse row: just the non-zero values, their column
numbers, and where each row starts).

## 2. Degrees

The **degree** of a node is how many edges touch it: a row sum of A,
or `A @ ones`.

```
  degree = A @ 1: min / median / max             -> 1 / 5 / 9
  sum of degrees = 2 x edges                     -> 386 = 2 x 193
  degree histogram (1, 2, ..., 8+)               -> 3 10 12 16 21 5 11 6
  busiest station by degree                      -> 5 (9 streets)
  mean neighbour degree vs mean degree           -> 5.04 vs 4.60
```

Every edge has two ends, so degrees always sum to twice the number of
edges, a good sanity check on any adjacency matrix you build. The
histogram is narrow because streets are local: nobody has 200 streets.
Social networks look very different, with a long tail of hubs, and
lesson 015's popularity curve is the same shape for the same reason.

The last line is the **friendship paradox**: on average, your
neighbours have more neighbours than you do. Not a quirk of this city:
busy nodes appear in many neighbour lists, so they get counted more
often. It's why sampling "a friend of a random person" finds hubs
faster than sampling a random person, an old trick in epidemiology.

## 3. Paths

A **path** is a sequence of edges from one node to another. The
**shortest path** in hops comes from **breadth-first search** (BFS):
visit all neighbours, then their neighbours, and so on; the first time
you reach a node is the shortest route. With travel times, the same
job is done by **Dijkstra's algorithm**, which always expands the
closest unfinished node.

```
  hand BFS agrees with scipy (all pairs)         -> True
  connected components (sizes)                   -> 2 (78, 6)
  main component: diameter / mean hops           -> 14 / 5.62
  main component: longest / mean trip (minutes)  -> 59 / 25.1
  mean trip west -> east vs within west (min)    -> 30.9 vs 18.9
```

- **Connected components**: the six suburb stations can't be reached
  from the city at all. Their distance to everything else is infinity,
  so any "average distance" has to say which component it's about.
  Always check components before computing anything global.
- **Diameter** is the longest shortest path: 14 hops, 59 minutes.
- Crossing the river adds 12 minutes on average, because only three
  bridges carry every west-east trip. Remember those bridges.

Powers of A count **walks** (paths allowed to revisit nodes):
(A^k)[i, j] is the number of k-step walks from i to j.

```
  walks of length 2 / 3 from station 0 to itself -> 5 / 14
  (A @ A)[i, i] = degree of i, for every i       -> True
  triangles in the network = trace(A^3) / 6      -> 166
```

A 2-step walk home is "out along one street and back", so the diagonal
of A^2 is the degree. A 3-step walk home is a **triangle**, and each
triangle is counted six times (3 starting corners x 2 directions).
Graph neural networks in lesson 019 are built on exactly this: one
multiplication by A means "look one step out".

## 4. Centrality: four ways to be important

Which stations matter most? It depends on what "matter" means, and
each centrality answers a different question. We compute them on the
78-station main component.

- **Degree**: how many streets meet here.
- **Closeness**: 1 / (average hops to everyone else). Can you get
  anywhere quickly from here?
- **Betweenness**: the share of all shortest paths between *other*
  pairs of stations that pass through this one. Would traffic jam
  here? Computed with **Brandes' algorithm**: one BFS per source, then
  a backwards sweep that splits the credit for each path.
- **Eigenvector centrality**: you're important if your neighbours are
  important. The answer is the top eigenvector of A, found by **power
  iteration** (multiply by A, normalise, repeat).
- **PageRank**: a random rider who follows a random street 85% of the
  time and teleports to a random station 15% of the time. Where do they
  spend their time? Google's original ranking, and a cousin of
  eigenvector centrality that behaves well on directed graphs.

```
  check: sum of betweenness = sum over pairs of (hops - 1) -> 13877.0 = 13877.0
  eigenvector: power iteration vs numpy (top value) -> 6.8755 vs 6.8755
  PageRank sums to 1 / corr with degree          -> 1.000 / 0.87
  top 5 by degree                                -> 5, 29, 27, 11, 63   (1 at a bridge)
  top 5 by closeness                             -> 33, 22, 36, 24, 63   (3 at a bridge)
  top 5 by betweenness                           -> 33, 63, 22, 6, 44   (3 at a bridge)
  top 5 by eigenvector                           -> 5, 29, 27, 19, 35   (0 at a bridge)
  top 5 by PageRank                              -> 63, 27, 11, 29, 5   (1 at a bridge)
  bridge stations: mean degree rank / betweenness rank -> 27 / 7 of 78
  rank corr: degree~PageRank, degree~betweenness -> 0.86, 0.56
  rank corr: eigenvector~closeness               -> 0.58
```

The first line is a neat check on the hand-written betweenness: a
shortest path of h hops has h - 1 stations in the middle, so the total
betweenness credit must equal the sum of (hops - 1) over all pairs. It
does, exactly. (The lesson's numbers also match `networkx`, if you
have it installed.)

Now the story. **Degree, eigenvector and PageRank** crown the busy
junctions in the dense western neighbourhood: station 5, with nine
streets. **Closeness and betweenness** crown station 33, which has
only six streets but sits at the western end of the southern bridge.
The six bridge stations rank 27th on average by degree but **7th** by
betweenness. Look at the right panel of `network.png`: the circled
bridge stations sit high above their neighbours in degree.

The striking case is eigenvector centrality. Station 33's score is
0.005, against 0.094 for station 5. The top eigenvector piles up in the
densest cluster (many walks loop around inside it) and almost ignores
the stations that hold the city together. If you wanted to know where a
broken dock would strand the most riders, eigenvector centrality would
point you at the wrong place. **Choose the centrality that matches your
question.**

## 5. Using the graph in a model

The simplest way to put a graph into a model is to give each node its
**neighbours' average features**: `D^-1 A X`, where D is the diagonal
matrix of degrees. That's one line, and it's the core operation of
the graph models in lesson 019.

```
  corr(own, neighbours' mean): docks             -> -0.09
  corr(own, neighbours' mean): height_m          -> 0.92
  corr(own, neighbours' mean): commute_share     -> 0.55
  corr(own, neighbours' mean): near_rail         -> 0.75
```

Height and nearness to the railway are strongly shared with neighbours
(they're geography), commute share somewhat (it follows the railway),
and dock counts not at all (the operator picked them at random). This
property, neighbours being alike, is called **homophily**, and it's
what makes graph features useful. When it's absent, as for docks,
averaging over neighbours just adds noise.

One more rule, and it's the most important one. Node numbers are
**arbitrary labels**. Relabel the stations and A becomes P A P^T for a
permutation matrix P, X becomes P X, and every honest computation
should just come out permuted:

```
  relabel stations: degrees just get permuted    -> True
  relabel: neighbour means just get permuted     -> True
  relabel: eigenvalues unchanged                 -> True
  relabel: row 0 of A as a feature vector        -> was 5 ones at [2 7 8]..., now at [ 0 47 65]...
```

Degrees, neighbour averages, eigenvalues and all the centralities
respect the relabelling. The tempting shortcut doesn't: feeding a
node's row of A to an ordinary model as 84 features. Renumber the
stations and the same street pattern lands in different columns, and
the model's weights mean nothing. It also can't handle a new station
(column 85 didn't exist when you trained). Graph methods are built
from operations like `A @ X` precisely because they're **permutation
equivariant**.

## What to remember

- A graph is **nodes + edges**. Hold it as an **adjacency matrix** A
  (sparse, in practice) and a **feature matrix** X.
- **Degree** is a row sum of A. Degrees sum to twice the edges. Your
  neighbours usually have more neighbours than you.
- **BFS** gives hop distances, **Dijkstra** weighted ones. Check
  **connected components** first.
- **(A^k)[i, j]** counts k-step walks; multiplying by A is "look one
  step out".
- Centralities answer different questions: **degree** (local busyness),
  **closeness** (reach), **betweenness** (bottlenecks), **eigenvector**
  and **PageRank** (being connected to the well connected).
- `D^-1 A X` averages neighbours' features. It helps when neighbours
  are alike (**homophily**).
- Node numbers are labels. Use operations that don't care about them.

## What to do now

1. Run `lesson.py` and look at `output/network.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 019 makes this predictive: guess a station's label from its
   neighbourhood, propagate labels across the graph, and learn a small
   node embedding.
