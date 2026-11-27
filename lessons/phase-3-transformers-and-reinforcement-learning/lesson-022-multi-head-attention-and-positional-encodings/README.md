# Lesson 022 - Multi-head attention and positional encodings

**Phase 3 - Transformers, efficient training and reinforcement learning** | Week 3, Day 1 | Friday 2026-11-27

> **Goal:** Build multi-head attention and add positions so order is
> not thrown away.

Time: about 75 minutes to read and run, plus the exercises. Nothing new
to install.

---

Lesson 021's dispatcher learned to find the right report in a stream,
and left us with two problems. One softmax gives **one set of
weights**, so a query can only look for one thing at a time. And
attention is **blind to order**: shuffle the stream and the answer
doesn't change.

Today we fix both, which gives us the two ideas that, together with
the feed-forward layer of lesson 023, make up a Transformer layer.
**Multi-head attention** runs several small attentions side by side,
each free to look for something different. **Positional encodings**
stamp each token with where it sits, so "the fourth report" means
something. We write both in NumPy, test them on questions that need
them, check the properties that make sinusoidal codes work, and end
with rotary encodings (RoPE), which many recent language models use.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-3-transformers-and-reinforcement-learning/lesson-022-multi-head-attention-and-positional-encodings/lesson.py
```

It takes a few seconds and saves `output/heads_and_positions.png`:
training curves for one head against two, what each of the two heads
attends to, the sinusoidal code as a picture, and how similar
positions are to each other.

## 1. Multi-head attention is a reshape

The recipe: project the tokens to Q, K and V as before (d numbers
each), **split** those d numbers into h groups of d/h, run attention
separately in every group, glue the h outputs back together, and mix
them with one more matrix, Wo.

In NumPy, the split is a `reshape` and a `transpose`, from (n, d) to
(h, n, d/h), and lesson 021's `attention` function handles the extra
leading axis without any change. We check it against the slow way,
h separate attentions on column slices:

```
  h=1: weights shape / params / max |fast - slow|      -> (1, 7, 7) / 1024 / 0.0e+00
  h=2: weights shape / params / max |fast - slow|      -> (2, 7, 7) / 1024 / 0.0e+00
  h=4: weights shape / params / max |fast - slow|      -> (4, 7, 7) / 1024 / 6.7e-16
  h=8: weights shape / params / max |fast - slow|      -> (8, 7, 7) / 1024 / 6.7e-16
```

Three things to notice:

- **The parameter count doesn't change with h.** It's always the four
  d x d matrices Wq, Wk, Wv, Wo: 4 x 16 x 16 = 1,024 here. More heads
  means more, narrower heads, not a bigger model.
- **What does grow is the weight tables**: h of them, each n x n.
  Eight heads give eight different ways of looking at the same
  sequence (and eight times lesson 021's quadratic memory).
- **The causal mask is shared**: one (n, n) mask broadcasts across
  every head.

## 2. Two lookups in one question

Why bother splitting? Give the dispatcher a question that needs two
lookups at once: "how many bikes at the market, and how many at the
hospital?" The question is 32 numbers, the two station identities
each in their own hidden "language", and the answer is two numbers.

With **one head** there's one set of weights, so both answers have to
be the *same* weighted average of the stream. With **two heads**, each
learns its own matrix and its own weights. Same training as yesterday,
600 Adam steps from zero, hand-written gradients:

```
  MAE: average of the stream, for both answers         -> 4.92 bikes
  MAE: one head, learned                               -> 3.64 bikes
  MAE: two heads, learned                              -> 0.43 bikes
  best any single weighting can do: mean |A - B| / 2   -> 3.43 bikes
  one head: mean weight on its top two reports         -> 0.52 + 0.29
```

- **One head is stuck**, and we can say exactly where. If both answers
  must be the same number, the best it can do is the midpoint of A and
  B, off by |A - B| / 2 on each: 3.43 bikes. The single head learned
  to do roughly that, splitting its weight across both stations (0.52
  and 0.29 on its top two), and lands at 3.64. Training longer won't
  help: the top-left curve in the figure has flattened.
- **Two heads**: 0.43 bikes, the same quality as yesterday's single
  lookup. The top-right panel shows why: head 1 puts its weight on one
  report, head 2 on another. Nobody told them which station to
  handle; the split fell out of the gradients.

In a real Transformer the heads aren't told what to look for either.
Trained language models end up with heads that track the previous
token, heads that match brackets, heads that copy names from earlier
in the text. Many heads also turn out to be redundant: you can
remove them and barely notice, which matters when we shrink models in
lesson 026.

## 3. Which report? Attention needs positions

Now a question that is *only* about order: "what did report number t
say?" The query is a code for the position t, nothing else.

Without positions, the keys are just station identities, and nothing
in them says where a report sits. To fix that, we **add** a position
code to every key: key_i = content_i + PE(i). The code is the classic
sinusoidal one from the original Transformer paper (Vaswani et al.,
2017):

    PE(p)[2i]   = sin(p / base^(2i/d))
    PE(p)[2i+1] = cos(p / base^(2i/d))

pairs of waves, from fast (the first dimensions) to slow (the last).
The paper uses base 10,000 for sequences of thousands of tokens; our
streams are 12 long, so we use 100. Streams of 12 reports, 600 Adam
steps:

```
  MAE: no positions                                    -> 5.09 bikes
  weight on report t: no positions                     -> 0.082
  MAE: with sinusoidal positions                       -> 1.36 bikes
  weight on report t: with sinusoidal positions        -> 0.790
  MAE: average of the stream                           -> 4.84 bikes
  same M, streams of 24, asking about reports 12-23    -> MAE 5.99 bikes, weight on t 0.091
```

- **No positions: hopeless.** 5.09 bikes, worse than averaging, with
  8.2% of the weight on the right report, which is the 1 in 12 of pure
  guessing. Exactly what lesson 021's shuffle test predicted.
- **With positions: 1.36 bikes**, 79% of the weight on report t. Not
  as sharp as the content lookup (0.48 yesterday), because the station
  identity added into each key gets in the way of the position code.
  Exercise 4 tries keeping them in separate columns.
- **The trap: positions never seen in training.** Ask about reports 12
  to 23 in longer streams and it collapses to guessing (0.091 of the
  weight on t). You'll often read that sinusoidal codes "extrapolate
  to longer sequences". The code itself is defined for any position,
  but what the model *learned to do* with it was only ever tested on
  0 to 11. Exercise 5 measures what helps.

## 4. What makes sinusoids a good position code

Why waves, and not just the number p? A few properties, checked
numerically:

```
  norm of every PE(p) (min / max)                      -> 2.8284 / 2.8284
  PE(3).PE(7), PE(50).PE(54), PE(120).PE(124)          -> 3.6627, 3.6627, 3.6627
  PE(p).PE(p+k) for k = 0, 1, 2, 5, 10, 40             -> [ 8.     7.314  5.73   3.779  2.066 -0.094]
  max |PE(p + 5) - T_5 PE(p)| over p = 0..194          -> 1.2e-14
```

- **Every position has the same size**, sqrt(d/2) = 2.83: each sin/cos
  pair has length 1. Position 1,000 doesn't shout louder than
  position 1, which a raw p would.
- **Similarity depends only on distance.** Positions 3 and 7, 50 and
  54, 120 and 124 all have the same dot product, 3.6627, because
  sin(a)sin(b) + cos(a)cos(b) = cos(a - b). It falls as positions move
  apart (8.0, 7.3, 5.7, ...), and the bottom-right panel shows the
  peak at the same position. That's what lets a query find "report
  t".
- **Shifting is a rotation.** For any offset k there's one fixed
  matrix T_k (a rotation in each sin/cos pair) with PE(p + k) = T_k
  PE(p) for *every* p, here exact to 1e-14. So a head can learn "look
  k back" as a single linear map, the same one everywhere in the
  sequence. That was the paper's stated reason for choosing sinusoids.

Many models instead **learn** a position table, one vector per
position (BERT and GPT-2 did). It works as well inside the trained
range, and has nothing at all to say about positions beyond it.

## 5. Rotary position encodings (RoPE)

Adding positions to the content has a flaw you saw in section 3: the
score q.k now mixes content and position, and depends on *absolute*
positions. RoPE (Su et al., 2021, now used in Llama, Mistral and many
others) does something neater: don't add anything, **rotate** the
query and the key. Each pair of numbers is rotated by an angle
proportional to the position, with the same fast-to-slow frequencies
as the sinusoids.

Because rotations compose, rotating q by m and k by n gives a score
that depends only on n - m:

```
  RoPE score, query at m, key at m-3, m=3/10/100       -> [-2.165 -2.165 -2.165]
  added-PE score, same pairs                           -> [-9.183 -1.255  5.014]
  RoPE keeps lengths: |rope(q, 37)| - |q|              -> 4.4e-16
```

- **RoPE: the same score**, -2.165, whether the pair sits at positions
  (3, 0), (10, 7) or (100, 97). Attention sees *how far back*, not
  *where*. That's usually what language needs: "the word just before"
  means the same thing on page 1 and page 30.
- **Added codes: three different scores** for the same content at the
  same distance. The model has to learn to undo that.
- **Rotations don't change lengths**, so RoPE doesn't disturb the
  scale of the scores (lesson 021, section 2).

RoPE is applied inside every attention layer, to Q and K only (values
are left alone), and it's a few lines: see `rope` in `lesson.py`.

## What to remember

- **Multi-head attention** = split d into h heads of d/h, attend in
  each, concatenate, mix with Wo. Same parameters for any h, h weight
  tables.
- **One head has one weighting per query.** A question that needs two
  lookups needs two heads, and the single-head error has a floor you
  can compute.
- Attention without positions can't tell reports apart by order.
  **Add a position code** (sinusoidal or learned) to the inputs, or
  **rotate** queries and keys (RoPE).
- **Sinusoids**: constant length, similarity depends only on distance,
  and a shift is a fixed rotation.
- **RoPE** makes the score depend on relative position only.
- **Length extrapolation is not free**: test on positions longer than
  you trained on before you trust it.

## What to do now

1. Run `lesson.py` and look at `output/heads_and_positions.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. On Monday, lesson 023 puts attention, residual connections, layer
   norm and a feed-forward layer together into one Transformer block
   you can step through, and lesson 024 trains a tiny language model
   end to end.
