# Lesson 001 - Why this track exists, and how we'll learn

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 1 | Friday 2026-11-13

> **Goal:** Connect the first course to the work ahead: uncertainty,
> causes, systems, and models you can stand behind.

Time: about 30 minutes to read and run, plus the exercises. Nothing new
to install if you finished the first course.

---

Welcome to the advanced track.

If you've finished [Your ML / AI Tutor](https://github.com/Moruri/Your-ML-AI-Tutor),
you can already load data, train a model, evaluate it honestly and ship
it with a model card. That is real skill. But you'll have noticed that
the hardest questions in that course were never "which model?" They
were the ones around it:

- **How sure are we?** A score of 0.30 means something very different
  after 10 rides than after 1,000.
- **What caused what?** A model can predict who stays brilliantly and
  still tell you nothing about what would make people stay.
- **Is it worth it?** A model that "catches more" can be worth less,
  once you count what its mistakes cost.

The first course taught you to *ask* those questions. This track teaches
you to *answer* them, and then goes on to the systems where they matter
most: forecasts, recommenders, graphs, transformers you build yourself,
reinforcement learning, and the insides of large language models.

Today there's no new technique. There are three small simulations, one
for each question, and a check that your setup is ready.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-001-why-this-track-exists-and-how-we-learn/lesson.py
```

It takes a second or two. Everything is simulated from a fixed seed, so
your numbers will match the ones below.

If you're starting fresh, make a venv and install the shared
requirements first:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Phase 1 only needs `numpy`, `pandas`, `scipy`, `matplotlib` and
`scikit-learn`. You can leave `torch` and `transformers` for later.

## 1. How sure are we?

Two new bike stations both have a 30% late-return rate. A tidy rule
says: close any station above 25%.

```
  Station A (10 rides): late rate, 95% bootstrap -> 0.30  [0.00, 0.60]
  Station B (1,000 rides): late rate, 95% bootstrap -> 0.30  [0.27, 0.33]
  naive rule closes both?                        -> yes, both are 0.30
  but could A really be below 25%?               -> easily: its interval reaches 0.00
```

Same number, very different evidence. Station B really is late a lot.
Station A has had three late returns out of ten; it might be fine and
just unlucky. You met the bootstrap in the first course, and it
already tells the story: A's interval runs from "never late" to "late
more often than not".

The bootstrap gets awkward exactly where we need help most, with tiny
samples and with prior knowledge ("most stations run at about 10%").
Lesson 002 starts the Bayesian way of thinking, which handles both
naturally: a belief is a whole distribution, and evidence moves it.

## 2. What caused what?

Simulated riders. A hidden trait, engagement, makes riders more likely
to switch on ride reminders *and* more likely to keep riding. The
reminders themselves do nothing. We built the data that way, so we
know the true effect is zero.

```
  stay rate, reminders on                        -> 0.731
  stay rate, reminders off                       -> 0.528
  observed gap (looks like an effect)            -> +0.203
  randomised gap (the real effect)               -> +0.002
  gap inside 'engaged' riders                    -> +0.007
  gap inside 'not engaged' riders                -> -0.008
```

Riders with reminders on stay 20 points more often. A classifier would
happily use "reminders on" as a strong feature, and for *predicting*
who stays it's a fine one. But if a product lead reads that model and
switches reminders on for everyone, nothing happens. When we randomise
who gets reminders, the gap shrinks to noise. Comparing within each
engagement group tells the same story.

This is the gap between **prediction** and **intervention**, and it's
the heart of lessons 006 to 009. You'll learn to write the question
down properly (potential outcomes), to design the experiment (A/B
tests), to draw the hidden cause (causal graphs), and to get a decent
answer when you can't experiment at all.

## 3. Is it worth it?

Two fraud flaggers for M-Pesa top-ups, scored on 10,000 simulated
top-ups with 1% fraud. Catching a fraud saves KES 2,000; a false alarm
costs about KES 150 in someone's time and an annoyed customer.

```
  simple rules: caught / false alarms            -> 79 / 200
  simple rules: accuracy, net value (KES)        -> 0.977, 128,000
  big model: caught / false alarms               -> 83 / 578
  big model: accuracy, net value (KES)           -> 0.940, 79,300
```

The pitch for the big model is "it catches more fraud", and it does:
83 against 79. It's also worth KES 48,700 less, because it nearly
triples the false alarms. Notice too how small 83 against 79 is: with
only about a hundred frauds in the sample, a gap of four is well within
the noise from section 1.

The rest of this track keeps coming back to this habit: decide what a
result is *worth* and how sure you are of it before deciding it's
better. It's what lets you say no to a more complicated system with a
clear conscience, or yes with evidence.

## 4. Is your environment ready?

The last section prints your Python and package versions. Ours were:

```
  python                                         -> 3.13.5
  numpy                                          -> 2.5.4
  pandas                                         -> 2.3.3
  scipy                                          -> 1.18.1
  matplotlib                                     -> 3.11.2
  sklearn                                        -> 1.9.1
```

Yours can differ as long as they sit inside the ranges in
`requirements.txt`. If anything says `MISSING`, install the
requirements and run it again.

## How this track works

The rhythm is the same as the first course:

- **Two lessons every weekday**, in strict order. Each has one goal.
- **Each lesson is a folder** with a `README.md` (read this), a
  `lesson.py` (run this, then change it), and `exercises.md` (do
  these). Some have a small data file.
- **Everything runs on a laptop CPU.** When a lesson needs something
  heavier, it will say so and keep the model small.
- **Predict, then run.** Before each script section, guess what it will
  print. Being wrong is where the learning is.

The track has five phases, roughly a week each:

1. **Bayesian thinking, causality and experiments** (001-010). How sure
   are we, and what caused what?
2. **Time series, recommenders and graphs** (011-020). Data with order,
   preference and structure, and the metrics they need.
3. **Transformers, efficient training and reinforcement learning**
   (021-032). Build one you can step through; train it without a data
   centre.
4. **LLM internals, fine-tuning, agents and advanced RAG** (033-044).
   Tokenisers, sampling, the KV cache, LoRA, careful agents, retrieval
   you can evaluate.
5. **Large-scale data and the advanced capstone** (045-050). Work with
   data that doesn't fit in memory, then scope, build and ship your own
   project.

The full plan is in [CURRICULUM.md](../../../curriculum/CURRICULUM.md).

## What to do now

1. Run `lesson.py` and read each section next to its output.
2. Do the exercises in [`exercises.md`](exercises.md). The hands-on one
   asks you to break section 2 on purpose.
3. Then go straight on to lesson 002, where a belief becomes a
   distribution.
