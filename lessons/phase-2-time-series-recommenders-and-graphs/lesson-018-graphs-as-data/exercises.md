# Lesson 018 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_018
cp lessons/phase-2-time-series-recommenders-and-graphs/lesson-018-graphs-as-data/lesson.py my_work_018/
cd my_work_018
python lesson.py
```

---

## 1. A graph you can hold in your head

Four stations, undirected streets: 0-1, 1-2, 2-0 and 2-3.

a) Write down A and the degrees. Check the sum.
b) What is (A^2)[0, 0]? How many triangles are there, and what is
   trace(A^3)?
c) Betweenness of every node, and closeness of nodes 2 and 3.

<details>
<summary>Check yourself</summary>

a)

    A = [[0, 1, 1, 0],
         [1, 0, 1, 0],
         [1, 1, 0, 1],
         [0, 0, 1, 0]]

Degrees 2, 2, 3, 1. Sum 8 = 2 x 4 edges.

b) (A^2)[0, 0] = degree of 0 = **2** (out to 1 and back, out to 2 and
back). One triangle (0, 1, 2), so trace(A^3) = **6**: each of its 3
corners, in 2 directions.

c) The only shortest paths with a middle station are 0-2-3 and 1-2-3,
both through 2. Betweenness: node 2 = **2**, all others **0**.
Closeness of 2: distances 1, 1, 1, so 3 / 3 = **1.0**. Node 3:
distances 2, 2, 1, so 3 / 5 = **0.6**. Check the identity from
section 4: pairs with 2 hops are (0,3) and (1,3), each contributing
2 - 1 = 1, total 2 = the betweenness sum.

</details>

## 2. One-way streets

The city makes some streets one-way, so A is no longer symmetric
(A[i, j] = 1 means you can ride from i to j).

a) What do the row sums and the column sums of A now count?
b) Is "sum of degrees = 2 x edges" still true? What replaces it?
c) A station with streets in but none out is a **dead end** for
   PageRank's random rider. What goes wrong, and how does the 15%
   teleport help?

<details>
<summary>Check yourself</summary>

a) Row sums: **out-degree** (streets leaving). Column sums:
**in-degree** (streets arriving).

b) Each directed edge has one tail and one head, so **sum of
out-degrees = sum of in-degrees = number of edges**.

c) Its column in the transition matrix is all zeros, so probability
that walks in never walks out: the total leaks away with each step and
no longer sums to 1. The usual fix is to make a dead end teleport to a
random station with probability 1. The 15% teleport handles a related
problem: without it, the random rider can get trapped in a cluster of
stations with no way out, and that cluster soaks up all the PageRank.

</details>

## 3. Which centrality?

For each question, pick the centrality you'd use and say why.

a) Where should the repair van be parked to reach any broken dock
   fastest?
b) If one station closes for roadworks, which closure would lengthen
   the most trips?
c) Which station is in the thick of the busiest neighbourhood?

<details>
<summary>Check yourself</summary>

a) **Closeness** (on travel-time distances, ideally): smallest average
time to everyone.

b) **Betweenness**: it measures how many shortest paths depend on the
station. In the lesson, that's station 33 at the bridge, not the
high-degree station 5. (Strictly, "how much longer trips get" is
answered by removing the node and recomputing, as in exercise 4;
betweenness is the cheap proxy.)

c) **Eigenvector** centrality (or degree): it rewards being connected
to well-connected stations, which is exactly "in a dense cluster".

</details>

## 4. Hands-on: close a bridge

In your copy, after `T` is built, set the travel time of one bridge to
zero (which removes it), rebuild `A = (T > 0).astype(int)` and rerun
the path and centrality sections. The bridges are the edges (4, 44),
(24, 42) and (33, 63). Try each one.

Predict first: which closure hurts most, and who becomes the new
bottleneck?

<details>
<summary>Check yourself</summary>

Mean west-to-east trip was 30.9 minutes:

    close (4, 44)   -> mean 32.7 min, longest 63.2
    close (24, 42)  -> mean 32.0 min, longest 59.5
    close (33, 63)  -> mean 36.0 min, longest 68.8

Closing (33, 63) hurts most, as its two ends had the two highest
betweenness scores (1297 and 891). After closing it, the top three by
betweenness are 56, 24 and 42: the traffic shifts to the middle bridge
(24, 42), and station 42, which had only 3 streets and a betweenness of
260, now handles over 1,000.

That's the general pattern: betweenness tells you where the network is
fragile, and removing a high-betweenness node moves the load to the
next bottleneck rather than spreading it out.

</details>

## 5. Hands-on: connect the suburb

Add one new bike lane from suburb station 78 to station 43 (the
closest pair, 3.4 km apart) and rerun.

a) How many components are there now?
b) Station 43 had a betweenness of 0 (it was on the edge of the city).
   Predict its new rank.

<details>
<summary>Check yourself</summary>

a) **1**: the whole network is connected, so global numbers like the
diameter now cover all 84 stations.

b) Its betweenness jumps from 0 to **462**, 14th of 84. Every trip
between the six suburb stations and the 78 city stations must pass
through 43 and the new lane. A station's importance isn't a property of
the station; it's a property of its position, and one new edge can
change it a lot.

</details>

## 6. Optional puzzle: why `+ np.eye(nb)`?

The lesson runs power iteration on `Ab + I` instead of `Ab`. The
comment says it gives the same eigenvector and avoids flip-flopping.

a) Why does adding I keep the eigenvectors and shift every eigenvalue
   by exactly 1?
b) A graph is **bipartite** if its nodes split into two groups with
   edges only between groups (a grid of streets, for instance). Its
   eigenvalues come in pairs +lambda and -lambda. Why does plain power
   iteration fail to settle on such a graph, and why does the shift
   fix it?

<details>
<summary>Check yourself</summary>

a) If A v = lambda v, then (A + I) v = A v + v = (lambda + 1) v. Same
vector, eigenvalue plus one.

b) Power iteration converges to the eigenvector whose eigenvalue has
the largest **absolute value**. With +lambda and -lambda tied, the
-lambda part flips sign every step, so x keeps swinging between two
states and never settles. After the shift, the two become lambda + 1
and |1 - lambda| = lambda - 1, no longer tied, so the top one wins.
(Our street network has triangles, so it isn't bipartite, but the
shift costs nothing and makes the function safe on any graph.) The
lesson checks the answer against `numpy.linalg.eigvalsh`, a good habit
whenever you write an iterative method yourself.

</details>
