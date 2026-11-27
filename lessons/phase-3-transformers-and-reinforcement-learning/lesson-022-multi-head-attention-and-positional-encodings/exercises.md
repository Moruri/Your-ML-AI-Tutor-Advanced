# Lesson 022 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_022
cp lessons/phase-3-transformers-and-reinforcement-learning/lesson-022-multi-head-attention-and-positional-encodings/lesson.py my_work_022/
cd my_work_022
python lesson.py
```

---

## 1. Count the heads

A model has d = 512 and h = 8 heads, and reads sequences of 2,048
tokens.

a) How wide is each head?
b) How many weights do Wq, Wk, Wv and Wo hold together (ignore
   biases)? What changes if you use 16 heads?
c) How many attention weights does one layer compute for one sequence?
   And with 16 heads?

<details>
<summary>Check yourself</summary>

a) 512 / 8 = 64 numbers per head.

b) 4 x 512 x 512 = 1,048,576, about a million. With 16 heads: exactly
the same. Each head is now 32 wide, and the four matrices are
unchanged; only the reshape differs (section 1).

c) 8 x 2,048 x 2,048 = 33.6 million weights per layer, per sequence.
With 16 heads, 67.1 million: the parameters stay the same, but the
attention tables double. Heads are cheap to store and not free to run.

</details>

## 2. Why can't one head answer two questions?

In section 2, a single head scores 3.64 bikes and the floor is
"mean |A - B| / 2". A colleague says: "Just make the head wider, say
d = 64, and it'll manage."

a) Why is the floor |A - B| / 2 for each question?
b) Does a wider head help?
c) Without adding heads, what's the simplest way to get both answers?

<details>
<summary>Check yourself</summary>

a) With one head both answers are the same number, y. The best single
number for two targets A and B, measured by absolute error, is any
point between them, and then |y - A| + |y - B| = |A - B|, so the
average error per answer is |A - B| / 2. No training can beat that.

b) No. The width changes how well the head can score the keys, not the
fact that it produces **one** weighting and therefore **one** average
per query. The limit is in the shape of the computation, not in the
number of parameters.

c) Ask twice. Send the two questions as two separate queries, two rows
of Q, and each gets its own weighting. That's what self-attention does
anyway: every token is its own query. Multiple heads are for when
**one** token needs several lookups at the same time, which in
language is all the time (who is "she"? what was the verb? where does
this bracket close?).

</details>

## 3. Distance only

Show that PE(p) . PE(p + k) doesn't depend on p, using the
sinusoidal formula in section 3. Then say what this implies about the
score between a query PE(t) and a key PE(i) as t moves along a
sequence.

<details>
<summary>Check yourself</summary>

Write w_i = 1 / base^(2i/d). Each sin/cos pair contributes

    sin(p w_i) sin((p + k) w_i) + cos(p w_i) cos((p + k) w_i) = cos(k w_i)

by the identity cos(a - b) = cos a cos b + sin a sin b. Summing over
the pairs, PE(p) . PE(p + k) = sum_i cos(k w_i), which only depends
on k. At k = 0 every term is 1, so the sum is d/2 = 8, the peak in
section 4.

So a query PE(t) scores keys by how far they are from t, the same way
wherever t is: that's the peaked curve in the bottom-right panel of
the figure, shifted along. (Adding the content to the key spoils this a
little, which is exercise 4.)

</details>

## 4. Hands-on: add, or keep apart?

In section 3, the keys are content + PE(i), added together. Try
keeping them in separate columns instead (lesson 019 did the same with
a node's own and neighbour features):

```python
K = np.concatenate([station_vec[st], np.broadcast_to(sinusoidal(np.arange(length)), (N, length, DP))], 2)
```

in `position_batch`, and make the matrix in the section 3 loop
`M = np.zeros((DP, 2 * DP))`. (The "no positions" line no longer
means anything after this change; ignore it.)

a) Predict the MAE and the weight on report t.
b) Real Transformers *add* the codes. Why might they, if concatenation
   is cleaner?

<details>
<summary>Check yourself</summary>

    keys                       MAE (reports 0-11)   weight on report t   reports 12-23 (len 24)
    content + PE (added)       1.36                 0.790                5.99
    [content, PE] (separate)   0.52                 0.918                5.43

a) Much better: 0.52 bikes, 92% of the weight on report t, as good as
lesson 021's content lookup. With separate columns, the learned M can
read the position part and ignore the content entirely. Added
together, the content is noise sitting on top of the position code.
Extrapolation is still bad (5.43), as expected: concatenating doesn't
teach the model about positions it never saw.

b) Width. Concatenating makes every token wider, and everything after
it (each projection, each layer) costs more. In high dimensions the
interference is smaller than in our 16: random vectors in hundreds of
dimensions are nearly orthogonal, so the model can learn to keep
content and position in mostly separate directions inside one vector.
RoPE avoids the question by not adding anything at all.

</details>

## 5. Hands-on: what helps with longer streams?

Section 3 failed on positions 12-23. Try two fixes, one at a time:

a) Change `base=100.0` to `base=10000.0` in `sinusoidal` (and, to keep
   things consistent, in `shift_matrix` and `rope`).
b) Put the base back to 100 and train on longer streams instead:
   change the training batch in the section 3 loop to
   `position_batch(3000, 24, np.arange(24), rng, use_pe)`.

Before running, predict which helps more, and what each one costs.

<details>
<summary>Check yourself</summary>

    change                     MAE (reports 0-11)   reports 12-23: MAE   weight on t
    as in the lesson           1.36                 5.99                 0.091
    base 10,000                1.73                 4.53                 0.297
    train on streams of 24     1.62                 1.64                 0.742

a) A bigger base makes most of the waves slower. Over 24 positions,
the slow dimensions barely change, so positions 12-23 look somewhat
like the ones seen in training: a little extrapolation (weight 0.297
instead of 0.091). The price: nearby positions are harder to tell
apart, and in-range accuracy drops (1.73 against 1.36). The base
should match the lengths you care about.

b) Training on the lengths you'll be asked about wins by far (1.64,
74% of the weight on report t). It's slightly worse on short streams
(1.62 against 1.36) because the same M now has to separate 24
positions. That's the honest answer to "does it extrapolate?": train
on it, or measure it. Long-context models are usually trained on long
sequences at the end, or have their RoPE frequencies stretched and
then fine-tuned, for exactly this reason.

</details>

## 6. Optional puzzle: why RoPE only sees distance

RoPE rotates each pair of a query at position m by an angle m w_i, and
each pair of a key at position n by n w_i.

a) Write R(theta) for the 2 x 2 rotation by theta. Show that
   (R(a) x) . (R(b) y) = x . (R(b - a) y).
b) Conclude that the RoPE score depends on the positions only through
   n - m.
c) Is the RoPE score still a function of content? What would happen if
   you also rotated the values?

<details>
<summary>Check yourself</summary>

a) (R(a) x) . (R(b) y) = x^T R(a)^T R(b) y. A rotation's transpose is
its inverse, R(a)^T = R(-a), and rotations add up: R(-a) R(b) =
R(b - a). So it equals x^T R(b - a) y.

b) The full score is a sum of such pair terms, each with angle
(n - m) w_i. Nothing depends on m or n separately: it's the
-2.165, -2.165, -2.165 in section 5.

c) Yes: x and y are still the content (the projected query and key);
RoPE only changes how they're compared. Rotating the values too would
make the output depend on the key's absolute position, the thing
we've just removed from the scores, and the next layer would have to
undo it. That's why RoPE is applied to Q and K only.

</details>
