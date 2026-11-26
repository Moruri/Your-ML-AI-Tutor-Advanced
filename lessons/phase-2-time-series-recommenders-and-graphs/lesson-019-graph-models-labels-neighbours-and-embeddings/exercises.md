# Lesson 019 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_019
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-019-graph-models-labels-neighbours-and-embeddings/lesson.py my_work_019/
cd my_work_019
python lesson.py
```

---

## 1. Label propagation by hand

Five stations in a line: 0 - 1 - 2 - 3 - 4. Station 0 is surveyed as
office, station 4 as leisure. Use the simplest version: each step,
every *unsurveyed* station takes the average of its neighbours' label
scores, and surveyed stations keep their labels (this is the
"clamped" version of Zhu and Ghahramani, 2003).

a) Start the unsurveyed stations at (0, 0). Write the office and
   leisure scores of stations 1, 2 and 3 after one step and after two.
b) Where does it end up? Which label does station 2 get?
c) Now add a street 1 - 3. Does anything change for station 2?

<details>
<summary>Check yourself</summary>

Scores written as (office, leisure).

a) Step 1: station 1 averages 0 and 2: ((1 + 0)/2, 0) = (0.5, 0).
Station 2 averages 1 and 3, both still (0, 0): (0, 0). Station 3:
(0, 0.5). Step 2: station 1 = average of (1, 0) and (0, 0) = (0.5, 0);
station 2 = average of (0.5, 0) and (0, 0.5) = (0.25, 0.25); station 3
= (0, 0.5).

b) The fixed point is a straight line between the two ends: station 1
= (0.75, 0.25), station 2 = (0.5, 0.5), station 3 = (0.25, 0.75).
This is the **harmonic** solution: every unlabelled score is the
average of its neighbours. Station 2 is a perfect tie. Breaking the tie
needs more information, such as features or a prior.

c) No. By symmetry, 1 and 3 still mirror each other, so station 2 still
sits at (0.5, 0.5). Station 1 now has neighbours 0, 2 and 3 and settles
at (0.625, 0.375) instead of (0.75, 0.25): the shortcut pulls it towards
the leisure end.

</details>

## 2. Is the GCN cheating?

The GCN's forward pass multiplies by Ahat over **all 600** stations,
including the 540 we score it on. Their features flow into the
surveyed stations' predictions during training.

a) Is that leakage?
b) A new station opens next month. What do you have to do to predict
   its type with the GCN? With label propagation?

<details>
<summary>Check yourself</summary>

a) No, as long as only **features and edges** of the test stations are
used, never their **labels**. That information really is available at
prediction time: we know every station's ride pattern and the map,
and only the survey answers are scarce. This setting is called
**transductive**. It would be leakage if, say, a feature had been
computed from the surveyed labels of the test stations.

b) **GCN**: add the station and its streets to A, recompute Ahat and
run the forward pass with the trained W1 and W2. The weights transfer
because they act on features, not node identities; this is what makes
GCNs **inductive** in principle. **Label propagation** has no weights
at all: you rerun the propagation on the new graph. The same goes for
the **walk embedding**, which has to be recomputed, and whose axes can
rotate between runs, so a classifier trained on last month's embedding
can't be reused as is. That's a real cost of embedding-first pipelines.

</details>

## 3. Where does over-smoothing end?

Smoothing with Ahat over and over, the accuracy drifted down from
79.9% (2 hops) to 75.0% (20 hops).

a) On a connected graph, what does Ahat^k X look like for very large k?
b) Why doesn't the accuracy fall all the way to the majority baseline
   at 20 hops in this lesson?

<details>
<summary>Check yourself</summary>

a) Ahat's largest eigenvalue is 1, and its eigenvector is
proportional to sqrt(d + 1) (degree plus the self-loop). Every other
eigenvalue is smaller in size, so they fade as k grows. In the limit,
every column of X becomes a constant times sqrt(d + 1): each station's
features are the same up to its degree. All that's left to classify
on is degree.

b) Two reasons. 20 is not infinity: the second largest eigenvalue is
close to 1 on a graph made of loosely connected neighbourhoods, so
neighbourhood-level differences take a long time to fade, and those
correlate with type. And the graph has six components, each smoothing
towards its own constant, so component membership survives forever.

</details>

## 4. Hands-on: how many surveys do you need?

Change `K_LAB` to 20 and then 200 and rerun. Before you do, predict:
which method gains the most from more labels, and does the ranking
change at 20?

<details>
<summary>Check yourself</summary>

    surveyed   features only   label propagation   SGC (Ahat^2 X)
    20         60.4%           55.3%               72.5%
    60         66.8%           74.1%               79.9%
    200        69.7%           81.6%               81.5%

- **Label propagation** gains the most (55% to 82%). With 20 labels
  most stations are many hops from any surveyed one, and whole
  neighbourhoods (and components) get no label mass, so it falls
  *below* features-only. With 200 it catches up with SGC.
- **SGC** is the best choice when labels are scarce: smoothing the
  features uses all 600 stations' ride data, whatever was surveyed,
  so it needs fewer labels to fit four weights.
- **Features only** gains least: it's limited by how noisy the hints
  are, not by the number of labels.

The general lesson: the best graph method depends on how many labels
you have. Report results at more than one label budget.

</details>

## 5. Hands-on: two hops on the trip graph

On the heterophilous trip graph, a residential station connects to
offices, and those offices connect back to residential stations. So
**two-hop** neighbours might be alike again.

In your copy, after `At` is built in section 6, make the two-hop graph

```python
A2t = ((At @ At) > 0).astype(float)
np.fill_diagonal(A2t, 0)
```

and run label propagation and SGC on it (remember to normalise; any
row of zeros needs a degree of 1 to avoid dividing by zero). Measure
the homophily of A2t first. Predict: does it beat features-only?

<details>
<summary>Check yourself</summary>

    two-hop graph: 8180 edges, homophily 53.5% (one-hop was 5.1%)
    label propagation on A2t   -> 49.5%
    SGC on A2t                 -> 65.9%
    features only              -> 66.8%

Homophily jumps from 5% to 54%, so the idea is right: two-hop
neighbours are much more alike. But not enough. Leisure stations
connect to both other types, so office -> leisure -> office and
residential -> leisure -> office paths mix things up again, and the
two-hop graph is also much denser (8,180 edges from 1,521). Neither
method beats features-only. The kept-apart model from the lesson
(71.5%) is still the right tool: it lets the data decide what a
neighbour means, rather than us guessing which power of A happens to
be homophilous. (Mixing hops 1 and 2 as separate columns is the H2GCN
idea; try it.)

</details>

## 6. Optional puzzle: why the closed form works

Label propagation iterates F <- alpha S F + (1 - alpha) Y0, with
S = D^-1/2 A D^-1/2 and 0 < alpha < 1.

a) Show that if it converges, the limit is
   F = (1 - alpha)(I - alpha S)^-1 Y0.
b) Why does it converge? (Hint: what are the eigenvalues of S?)
c) What happens as alpha approaches 1, and what does that suggest
   about choosing alpha? Try alpha = 0.5, 0.9 and 0.99.

<details>
<summary>Check yourself</summary>

a) At a fixed point, F = alpha S F + (1 - alpha) Y0, so
(I - alpha S) F = (1 - alpha) Y0. If I - alpha S is invertible, F is
as stated.

b) S is symmetric and similar to the random-walk matrix D^-1 A, whose
eigenvalues lie between -1 and 1. So alpha S has eigenvalues of size
at most alpha < 1, I - alpha S is invertible, and the error shrinks by
at least a factor alpha per step. Unrolling the iteration gives a
geometric series, F = (1 - alpha) * sum over k of (alpha S)^k Y0: label
mass that travelled k hops is discounted by alpha^k.

c) As alpha -> 1, far-away labels count almost as much as near ones,
and the answer drifts towards whichever class dominates each
component, the label version of over-smoothing:

    alpha 0.5  -> 74.9%
    alpha 0.9  -> 74.1%
    alpha 0.99 -> 67.1%

Small alpha keeps things local. Here 0.5 and 0.9 are about the same
and 0.99 is clearly worse. Tune alpha like any other hyperparameter,
by validating on some held-back surveyed stations, never on the test
set.

</details>
