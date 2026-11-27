# Lesson 021 - Attention from first principles

**Phase 3 - Transformers, efficient training and reinforcement learning** | Week 3, Day 1 | Friday 2026-11-27

> **Goal:** Implement scaled dot-product attention in NumPy and show
> what a query attends to.

Time: about 70 minutes to read and run, plus the exercises. Nothing new
to install today. Phase 3 brings in `torch` later on, but this lesson
and the next are pure NumPy, so you can see every number.

---

Welcome to phase 3. You met Transformers in the first course, mostly
as something you load and call. This time we build one, a piece at a
time, small enough to step through. The piece everything else hangs on
is **attention**, and it's simpler than its reputation.

Here's the picture we'll use. A dispatcher at the depot hears a stream
of radio reports: "market station, 7 bikes", "hospital, 12 bikes", and
so on. Then a van driver asks: "how many at the one by the market?"
To answer, the dispatcher has to find the right report in the stream
and read off its number. That's a **lookup**: compare the question
with each report, pick the one that matches, return what it says.

Attention is that lookup, made smooth so you can train it with
gradients. Today we write it from scratch: a soft dictionary lookup,
the softmax and why we divide by sqrt(d), self-attention over a
sequence with a causal mask, and then a lookup the model has to
*learn*, with gradients written by hand and checked. We finish with
the two things attention can't do on its own, which set up lesson 022.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-3-transformers-and-reinforcement-learning/lesson-021-attention-from-first-principles/lesson.py
```

It takes a few seconds and saves `output/attention.png`: the soft
lookup at three sharpness levels, the effect of scaling, the learned
attention weights for twelve test questions, and the training curve.

## 1. A lookup you can differentiate

Four stations, each with a short description (the **key**, three
numbers) and a bike count (the **value**). The question (the
**query**) is "the one by the market", said a bit vaguely. We score
each key by its dot product with the query:

```
  dot-product scores (market, hosp, stad, lake)      -> [0.9 0.1 0.2 0.7]
  hard lookup: value of the best key                 -> 7.0
  soft lookup, sharpness  1.0: weights               -> [0.362 0.163 0.18  0.296]
  soft lookup, sharpness  1.0: answer                -> 7.69
  soft lookup, sharpness  4.0: weights               -> [0.645 0.026 0.039 0.29 ]
  soft lookup, sharpness  4.0: answer                -> 7.55
  soft lookup, sharpness 16.0: weights               -> [0.961 0.    0.    0.039]
  soft lookup, sharpness 16.0: answer                -> 7.08
```

A **hard lookup** takes the best key and returns its value: 7 bikes.
It's correct, but `argmax` has no useful gradient. Nudge the query and
the answer either doesn't move or jumps. You can't learn through it.

A **soft lookup** turns scores into weights with the **softmax**,

    a_i = exp(s_i) / sum_j exp(s_j)

and returns the weighted average of the values, sum_i a_i v_i. The
weights are positive and add up to 1, and they change smoothly with
the query, so gradients flow. Multiply the scores by a bigger number
and the softmax gets sharper, moving towards the hard lookup (7.08 at
sharpness 16). The lakeside station keeps a little weight because its
key really does look a bit like "market".

That's all attention is: **scores from query-key dot products, a
softmax, a weighted average of values.** In matrix form, for many
queries at once,

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

which is the `attention` function at the top of `lesson.py`, all six
lines of it.

## 2. Why divide by sqrt(d)

Where does the sqrt(d) come from? Take a query and ten keys with
random entries of variance 1 in d dimensions. Their dot product is a
sum of d terms, so its variance is d and its spread is sqrt(d):

```
  d=4    score std raw / scaled                      -> 2.0 / 1.00
  d=4    mean top weight raw / scaled                -> 0.492 / 0.305
  d=4    softmax gradient size raw / scaled          -> 0.3357 / 0.340
  d=64   score std raw / scaled                      -> 8.0 / 1.00
  d=64   mean top weight raw / scaled                -> 0.866 / 0.321
  d=64   softmax gradient size raw / scaled          -> 0.1595 / 0.347
  d=512  score std raw / scaled                      -> 22.5 / 1.00
  d=512  mean top weight raw / scaled                -> 0.955 / 0.319
  d=512  softmax gradient size raw / scaled          -> 0.0616 / 0.348
```

Without scaling, a bigger head makes the scores spread further, and
the softmax goes almost one-hot *before any learning has happened*: at
d = 512 the top key gets 95.5% of the weight, picked by chance. Worse,
the softmax's gradient (the size of its Jacobian, diag(a) - a a^T)
shrinks from 0.34 to 0.06. A saturated softmax barely learns.

Dividing by sqrt(d) puts the scores back to a spread of 1 whatever the
width, so the top weight stays near 0.32 and the gradient stays
healthy. It's the same idea as careful weight initialisation from the
first course: keep the signal at a sensible scale so training can
start. The model can still learn to be sharp; it just doesn't start
that way.

## 3. Self-attention over a sequence

In a Transformer, the queries, keys and values all come from the same
sequence. Each token (here, each report, eight numbers) is projected
three ways with learned matrices:

    Q = X Wq,   K = X Wk,   V = X Wv

and every token attends to every token, itself included. That's
**self-attention**.

```
  input / output shape                               -> (6, 8) / (6, 4)
  each row of weights sums to                        -> [1. 1. 1. 1. 1. 1.]
  causal: weight above the diagonal (max)            -> 0.0
  causal: report 0 attends to itself with weight     -> 1.0
  causal: report 5's weights                         -> [0.049 0.113 0.221 0.097 0.225 0.295]
  shuffle the reports: max |out(PX) - P out(X)|      -> 2.2e-16
```

- Six tokens in, six tokens out, each output a mix of the values. The
  weights form a 6 x 6 table, one row per query, each row adding to 1.
- **Causal mask.** A language model predicting the next word must not
  peek at it. We set the scores above the diagonal to minus infinity
  before the softmax, so their weights are exactly 0. Token 0 can only
  see itself (weight 1.0); token 5 can see all six. This is how
  lesson 024's language model will be trained on every position at
  once without cheating.
- **Shuffle the inputs and the outputs shuffle the same way**, to
  within 2e-16. Self-attention is *permutation equivariant*: it has no
  idea which report came first. Hold on to that; it's the problem
  lesson 022 solves.

## 4. Learning to look things up

Now a lookup the model has to learn. The city has 40 stations, each
with a 16-number identity. A stream holds 10 reports from different
stations: the key is the station's identity (plus a little noise), the
value is its bike count, 0 to 20. The question names one station in
the stream, but it arrives **in a different language**: the identity
rotated by a hidden matrix R. It's like the van drivers using
nicknames the dispatcher has to learn.

So comparing the query with the keys directly is useless. The model
gets one learnable matrix M and scores

    s_i = q^T M k_i / sqrt(16)

which is exactly what Wq and Wk do in a real head (M = Wq Wk^T). We
train M with Adam on 3,000 streams, with the gradient worked out by
hand. Through the weighted average and the softmax, the gradient with
respect to each score is

    dL/ds_i = a_i (v_i - answer) dL/d(answer)

Read it: a score goes up when its value is *above* the current answer
and the answer is too low. Before trusting it, we check it against
finite differences:

```
  gradient check: max relative error                 -> 5.3e-08
  MAE: average of all values (no lookup)             -> 4.91 bikes
  MAE: attention with M = identity (untrained)       -> 4.93 bikes
  MAE: attention with learned M                      -> 0.48 bikes
  MAE: hard argmax with learned M                    -> 0.004 bikes
  weight on the right report, mean (learned)         -> 0.916
  argmax lands on the right report                   -> 99.8%
  cosine(learned M, the hidden translation R)        -> 0.822
  same M, streams of 30 reports: MAE                 -> 1.18 bikes
  same M, 30 reports: weight on the right one        -> 0.793
```

- **Baselines first.** Averaging the stream is off by 4.91 bikes.
  Untrained attention (M = identity) is no better, 4.93: in the wrong
  language, the query matches nothing.
- **Learned M**: 0.48 bikes. On average 91.6% of the weight lands on
  the right report, and the top weight is on the right one in 99.8% of
  test questions. The learned M points mostly in the direction of the
  hidden rotation R (cosine 0.82). The model found the translation.
- Now look at the bottom-left of the figure: **what each query attends
  to.** Most rows are a single dark square on the red cross. A couple
  spread a little weight over look-alike stations. That's where the
  0.48 bikes of error comes from. With the same M, hard argmax gets
  0.004, because the remaining softness *blurs* the answer even when
  the right report is on top. Training longer sharpens M (the curve is
  still going down), which is exercise 4.
- **Same weights, longer streams.** Attention has no parameters tied
  to the length of the input, so the same M works on 30 reports. It
  gets worse (0.79 of the weight on the right one, 1.18 bikes) because
  there are more look-alikes sharing the softmax. Softmax attention
  gets *diluted* as context grows. Long-context models fight exactly
  this.

## 5. Blind to order, and quadratic in length

Two limits to carry into the next lessons.

```
  true answer for this question                      -> 0 bikes
  answer before / after shuffling the stream         -> 0.5579 / 0.5579
  score matrix,   1,000 tokens, float32, one head    -> 0.004 GB
  score matrix,   8,000 tokens, float32, one head    -> 0.256 GB
  score matrix,  32,000 tokens, float32, one head    -> 4.1 GB
  score matrix, 128,000 tokens, float32, one head    -> 65.5 GB
```

- **Order.** Shuffle the stream and the answer is identical, to every
  digit. For "bikes at the market?" that's what we want. For "what did
  the *last* report say?" or for language, where "dog bites man" isn't
  "man bites dog", it's fatal. Attention needs to be told about
  positions: lesson 022.
- **Cost.** Every query scores every key, so an n-token sequence makes
  an n x n table per head, per layer. Double the context, quadruple
  the table. At 32,000 tokens a single head's scores in float32 take
  4.1 GB. This is why long-context models use tricks such as computing
  attention in tiles without ever storing the table (FlashAttention),
  attending to a window, or caching keys and values when generating.
  We'll meet some of these in lesson 025 and phase 4.

## What to remember

- **Attention is a soft lookup**: scores from query-key dot products,
  a softmax, a weighted average of values.
  `softmax(Q K^T / sqrt(d)) V`.
- **Divide by sqrt(d)** so the scores have spread about 1 at any
  width; otherwise the softmax saturates at random and gradients
  vanish.
- **Self-attention** projects one sequence into Q, K and V. A **causal
  mask** (minus infinity above the diagonal) stops a token seeing the
  future.
- The gradient through a softmax-weighted average is
  `a_i (v_i - answer)`: check hand-written gradients numerically.
- To see **what a query attends to**, look at its row of weights. A
  soft row blurs the answer even when the top weight is right.
- Attention is **blind to order** and costs **n squared** in memory
  and time.

## What to do now

1. Run `lesson.py` and look at `output/attention.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 022 gives attention several heads, so it can look for more
   than one thing at once, and positions, so order stops being thrown
   away.
