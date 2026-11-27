# Lesson 021 - Exercises

Three quick checks, two hands-on experiments, and an optional puzzle.
Predict first, then run.

Activate your venv. For the hands-on tasks, copy `lesson.py` somewhere
of your own and edit the copy. From the repo root:

```bash
mkdir -p my_work_021
cp lessons/phase-3-transformers-and-reinforcement-learning/lesson-021-attention-from-first-principles/lesson.py my_work_021/
cd my_work_021
python lesson.py
```

---

## 1. A softmax by hand

Three keys score s = (2, 1, 0) against a query. Their values are
v = (10, 4, 0).

a) Compute the attention weights and the answer. (e is about 2.718.)
b) Add 100 to every score. What changes?
c) Double every score. What changes, and which way does the answer
   move?

<details>
<summary>Check yourself</summary>

a) exp(2), exp(1), exp(0) = 7.389, 2.718, 1, which add up to 11.107.
Weights: (0.665, 0.245, 0.090). Answer: 6.65 + 0.98 + 0 = 7.63.

b) Nothing. exp(s + c) = exp(c) exp(s), and exp(c) cancels between
top and bottom. That's why `softmax` in `lesson.py` subtracts the
maximum first: same answer, no overflow when scores are large.

c) Scores (4, 2, 0): exp = 54.6, 7.39, 1, weights (0.867, 0.117,
0.016), answer 9.14. Doubling the scores is like halving a
temperature: the softmax sharpens towards the top key, and the answer
moves towards its value, 10. Scaling *does* matter, which is the whole
point of section 2.

</details>

## 2. Why minus infinity?

The causal mask sets the forbidden scores to minus infinity before the
softmax. A colleague suggests setting them to 0 instead, "since zero
means no attention".

a) Is that right? Use the three scores from exercise 1 and mask the
   last two.
b) What about masking *after* the softmax: set the forbidden weights
   to 0 and leave the rest?

<details>
<summary>Check yourself</summary>

a) No. A score of 0 isn't a weight of 0: exp(0) = 1. Scores (2, 0, 0)
give weights (0.787, 0.107, 0.107), so the "masked" keys still get a
fifth of the weight between them, and a language model would see the
future. exp(minus infinity) = 0 is the only score that gives exactly
zero weight.

b) Then the row no longer adds up to 1: the weights are (0.665, 0, 0),
and the answer shrinks towards 0 for no reason. You'd have to
renormalise, and renormalising the surviving weights is exactly the
same as the minus-infinity softmax. Do it before, it's simpler.

</details>

## 3. Read the gradient

Section 4 says the gradient of the loss with respect to score i is

    dL/ds_i = a_i (v_i - answer) dL/d(answer)

The model answers 8 bikes and the truth is 12, so dL/d(answer) is
negative. The stream holds reports worth 12 (weight 0.5), 3 (weight
0.3) and 9 (weight 0.2).

a) Which scores does gradient descent push up, and which down?
b) Why does a report with weight 0.001 barely move, even if it's the
   right one?

<details>
<summary>Check yourself</summary>

a) The answer is too low. Gradient descent moves each score *against*
its gradient, so scores whose value is above the current answer go up
(12 and 9, since 12 - 8 and 9 - 8 are positive), and scores whose value
is below go down (3). The biggest push is on the 12: weight 0.5 times
a gap of 4. That's the right report.

b) The factor a_i. A report that gets almost no attention gets almost
no gradient: the model only learns about reports it already looks at.
This is another reason not to start with a saturated softmax
(section 2). If one random key gets 95% of the weight at the start,
the others are nearly frozen.

</details>

## 4. Hands-on: train longer, then make it harder

a) Change `range(1, 601)` in the training loop to `range(1, 1501)`.
   Predict the MAE and the weight on the right report.
b) Put it back to 600 steps, and make the reports and the questions
   much noisier: change both `rng.normal(0, 0.05, ...)` in
   `make_batch` to `rng.normal(0, 0.3, ...)`. Predict again.
c) With the noise at 0.3, train for 1500 steps. Better or worse?

<details>
<summary>Check yourself</summary>

    setting                  learned MAE   hard argmax MAE   weight on right   30 reports MAE
    600 steps, noise 0.05    0.48          0.004             0.916             1.18
    1500 steps, noise 0.05   0.22          0.000             0.962             0.62
    600 steps, noise 0.3     3.65          3.861             0.406             4.74
    1500 steps, noise 0.3    3.88          4.054             0.403             4.91

a) Longer training sharpens M: 96% of the weight on the right report,
and the MAE halves to 0.22. The longer streams gain most (1.18 to
0.62), because a sharper softmax is less diluted by look-alikes.

b) Each station's identity has length about 1, and noise of 0.3 in
each of 16 directions has length about 1.2. The reports drown in it.
The model still learns something (3.65 against 4.91 for averaging) but
finds the right report only 43% of the time. Notice that the hard
argmax is now *worse* than the soft answer (3.86 against 3.65): when
it isn't sure, hedging across the candidates is the better bet.

c) Worse on the test set (3.88), though the training loss keeps
falling. With signal this weak, the extra steps memorise the noise in
the 3,000 training streams. Same rule as always: hold data back to
choose when to stop.

</details>

## 5. Hands-on: break the scaling

In the training section, start M from a random matrix instead of
zeros: `M = np.random.default_rng(5).normal(size=(D, D))`. Print the
mean top weight at the start, `forward(M, Qte, Kte, Vte)[1].max(1).mean()`.
Then make a second copy with every `/ np.sqrt(D)` removed (in
`forward` and in `loss_and_grad`), and run both. Then change
`D, N_ST, L = 16, 40, 10` to `D, N_ST, L = 128, 40, 10` and run both
again.

a) Before running: which version learns better, and does the answer
   depend on D?
b) Run all four and compare the starting top weight, the training
   RMSE in `curve` at steps 50 and 600, and the test MAE.

<details>
<summary>Check yourself</summary>

    D     scaling   top weight at start   RMSE step 50   RMSE step 600   test MAE
    16    sqrt(D)   0.141                 3.96           0.98            0.48
    16    none      0.918                 3.82           0.01            0.00
    128   sqrt(D)   0.119                 1.52           0.16            0.15
    128   none      0.988                 5.91           2.84            0.84

A surprise first: at D = 16, **no scaling wins**. Why? Our identities
have length about 1, not entries of size 1, so q^T M k already has a
spread of about 1 with a random M, and dividing by 4 makes the start
*too* soft (0.141, close to the uniform 0.1). The unscaled version
starts sharp-ish and a lookup task likes sharp.

At D = 128 the picture flips. Unscaled, the random start is almost
one-hot (0.988 of the weight on one report, chosen by chance), most
reports get almost no gradient (exercise 3b), and after 600 steps it's
stuck at 2.84 RMSE. The scaled version barely notices the wider head
and does better than at D = 16.

The lesson: 1/sqrt(d) is not magic. It's a rule for keeping scores at
spread 1 *when the entries of q and k have size about 1*, which is
what a Transformer's initialisation and layer norm (lesson 023) aim
for. What really matters is the spread of the scores at the start,
and that's worth measuring rather than assuming.

</details>

## 6. Optional puzzle: the softmax-average gradient

The answer is y = sum_i a_i v_i with a = softmax(s).

a) Show that da_i/ds_j = a_i (delta_ij - a_j), where delta_ij is 1 if
   i = j and 0 otherwise.
b) Use it to show dy/ds_j = a_j (v_j - y).
c) Why do the dy/ds_j always add up to zero? What does that say about
   adding the same number to every score?

<details>
<summary>Check yourself</summary>

a) a_i = exp(s_i) / Z with Z = sum_k exp(s_k), and dZ/ds_j = exp(s_j).
By the quotient rule, da_i/ds_j = delta_ij exp(s_i) / Z -
exp(s_i) exp(s_j) / Z^2 = a_i delta_ij - a_i a_j.

b) dy/ds_j = sum_i v_i a_i (delta_ij - a_j) = a_j v_j - a_j sum_i a_i
v_i = a_j (v_j - y).

c) sum_j a_j (v_j - y) = y - y (sum_j a_j) = y - y = 0. Moving every
score up by the same amount is a step in the direction (1, 1, ..., 1),
and the change in y is the sum of the gradients: zero. That's
exercise 1b again, seen through calculus. The softmax only cares
about *differences* between scores.

</details>
