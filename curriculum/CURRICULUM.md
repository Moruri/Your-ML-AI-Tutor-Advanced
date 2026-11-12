# Curriculum map

This is the whole advanced track, start to finish. Everything is taught in
Python, and every lesson is meant to run on a laptop CPU.

It assumes you have finished [Your ML / AI Tutor](https://github.com/Moruri/Your-ML-AI-Tutor)
— the 84 lessons on Python, data science, classical ML, deep learning,
transformers, RAG, fine-tuning, evaluation, safety and MLOps — or that you
already have the equivalent grounding. We start at lesson **001** in this
repository; the numbering here is new.

**How to read this page**

- Lessons are numbered `001`, `002`, ... and taught in strict order. Each one
  builds on the one before it, so please don't skip ahead.
- We publish two lessons every weekday (Monday to Friday). Nothing on weekends.
  That's the day to rest, or to redo an exercise that didn't click.
- Each lesson has one job, written as a one-line goal. If you can do that thing
  when you finish, you're done. Nothing more is expected.
- Lessons **001** and **002** are where we start. See
  [PROGRESS.md](PROGRESS.md) for what's published so far.

A rough sense of pace: two lessons a day, five days a week, means fifty
lessons take five weeks of steady work. That's on purpose. The ideas here
need time to settle.

---

## Phase 1 - Bayesian thinking, causality and experiments

*Goal of the phase: go from "the model predicted X" to "here's how sure we
are, and whether X actually caused Y."*

Installs: the shared `requirements.txt` stack (`numpy`, `pandas`,
`matplotlib`, `scipy`, `scikit-learn`). No new packages required.

| # | Lesson | Goal |
|---|--------|------|
| **001** | **Why this track exists, and how we'll learn** (start here) | Connect the first course to the work ahead: uncertainty, causes, systems, and models you can stand behind. |
| **002** | **Probability as belief** (start here) | Represent a belief as a distribution and update it with Bayes' rule on a small, honest example. |
| 003 | Conjugate priors you can finish on paper | Update a Beta-Binomial model by hand and in code, and read the posterior as a statement of uncertainty. |
| 004 | Sampling from a posterior | Implement a Metropolis-Hastings sampler, plot the chain, and decide whether it has mixed. |
| 005 | Hierarchical models and partial pooling | Share statistical strength across groups so rare ones shrink and common ones stay sharp. |
| 006 | Potential outcomes and causal questions | Write a decision as a contrast of potential outcomes and see why a good predictor can still be a bad policy. |
| 007 | Randomised experiments and A/B tests | Design a simple experiment, estimate an average treatment effect, and report it with a confidence interval. |
| 008 | Confounding, graphs and what to adjust for | Draw a small causal graph, name a confounder, and adjust without opening a collider. |
| 009 | Quasi-experiments when you can't randomise | Estimate a treatment effect with difference-in-differences on a before/after series with a control. |
| 010 | Mini-project: decide with uncertainty | Analyse a simulated experiment both Bayesian-style and frequentist-style, and write a recommendation a product lead could use. |

## Phase 2 - Time series, recommenders and graphs

*Goal of the phase: model data that has order, preference or structure, and
evaluate it with the metrics those problems actually need.*

| # | Lesson | Goal |
|---|--------|------|
| 011 | Time series as a modelling problem | Decompose a series into trend, seasonality and remainder, and set a forecasting baseline that is hard to beat. |
| 012 | Autoregression, stationarity and holdouts that don't cheat | Fit an AR model, test for stationarity, and evaluate with a rolling origin instead of a random split. |
| 013 | Exponential smoothing and classic forecasters | Forecast with Holt-Winters and compare it fairly against a naive seasonal baseline. |
| 014 | Features, exogenous signals and when ML helps a forecast | Add calendar and lag features to a tabular model and see when it beats the classical methods. |
| 015 | Recommender systems: the problem and the traps | Frame recommendation as predict-and-rank, and measure it with ranking metrics instead of accuracy. |
| 016 | Collaborative filtering and matrix factorisation | Factor a user-item matrix, predict missing ratings, and discuss cold start. |
| 017 | Content, implicit feedback and ranking losses | Build a recommender from clicks rather than stars, and train it with a ranking objective. |
| 018 | Graphs as data | Represent a network as an adjacency and a feature matrix, and compute degrees, paths and simple centralities. |
| 019 | Graph models: labels, neighbours and embeddings | Predict a node label from its neighbourhood and learn a small graph embedding. |
| 020 | Mini-project: forecast, recommend or connect | Take one of the three threads to a short, evaluated write-up on a small public dataset. |

## Phase 3 - Transformers, efficient training and reinforcement learning

*Goal of the phase: build a Transformer you can step through, train it
without a data-centre, and understand what reinforcement learning is
actually optimising.*

Installs: `torch` (CPU is fine).

| # | Lesson | Goal |
|---|--------|------|
| 021 | Attention from first principles | Implement scaled dot-product attention in NumPy and show what a query attends to. |
| 022 | Multi-head attention and positional encodings | Build multi-head attention and add positions so order is not thrown away. |
| 023 | A Transformer block you can step through | Assemble attention, residual connections, layer norm and a feed-forward layer into one working block. |
| 024 | A tiny language model, trained end to end | Train a small character- or byte-level Transformer on a toy corpus and generate from it. |
| 025 | Making training efficient on a laptop | Use gradient accumulation, checkpointing and careful batching so a small model still learns on a CPU. |
| 026 | Quantisation, distillation and why size isn't destiny | Compress a small model and measure what you keep and what you lose. |
| 027 | What reinforcement learning is actually optimising | Write a Markov decision process, a return and a policy, and run a tiny grid-world by hand. |
| 028 | Value methods: Q-learning you can debug | Implement tabular Q-learning and watch the value function fill in. |
| 029 | Policy gradients and REINFORCE | Train a tiny policy with REINFORCE and see why variance is the enemy. |
| 030 | Actor-critic, reward design and failure modes | Add a baseline or a critic, watch an agent game a badly specified reward, and name problems that should not be framed as RL at all. |
| 031 | Exploration, generalisation and when to stop | Try simple exploration strategies, test whether the policy transfers, and know when to call the experiment done. |
| 032 | Mini-project: a small agent or a small Transformer | Train either a compact Transformer or a compact RL agent, evaluate it honestly, and write up the trade-offs. |

## Phase 4 - LLM internals, fine-tuning, agents and advanced RAG

*Goal of the phase: look inside a language model, adapt it carefully, and
build assistants whose retrieval and tool use you can actually evaluate.*

Installs: `transformers` (heavier; introduced as needed). CPU is still fine.

| # | Lesson | Goal |
|---|--------|------|
| 033 | Tokenisers: bytes, subwords and what gets lost | Inspect a small BPE tokeniser and see how tokenisation changes what a model can notice. |
| 034 | Sampling, temperature and the shape of a distribution | Compare greedy, temperature, top-k and nucleus sampling, and measure repetition and diversity. |
| 035 | KV cache and why inference is a memory problem | Implement a tiny key-value cache and show why incremental decoding is faster than recomputing attention. |
| 036 | Context, windows and long documents | Chunk, stride and position a long document, and see what falls out of a finite window. |
| 037 | Fine-tuning versus prompting, honestly | Fine-tune a small model on a narrow task and compare it to a carefully prompted baseline. |
| 038 | LoRA and parameter-efficient fine-tuning, by intuition | Freeze a base model, train a low-rank adapter, and count what actually moved. |
| 039 | Preference data and alignment at a small scale | Rank outputs, apply a simple preference update, and discuss what alignment can and cannot do. |
| 040 | Agents that call tools, built carefully | Give a model a few Python tools, loop until it stops, and log every call. |
| 041 | Where agent loops break | Reproduce a tool typo, an infinite loop and a hallucinated argument, and add the guards that catch them. |
| 042 | Advanced RAG: chunking, hybrid retrieval and reranking | Combine lexical and embedding search, rerank the hits, and show a case where naive RAG fails. |
| 043 | Evaluating retrieval and grounded generation | Build a small eval set and measure recall, faithfulness and answer quality separately. |
| 044 | Mini-project: a grounded assistant | Ship a tiny retrieve-then-generate (or tool-using) assistant with an evaluation report, not just a demo. |

## Phase 5 - Large-scale data and the advanced capstone

*Goal of the phase: handle data that does not fit in memory, then scope,
build and ship a project that uses what this track has taught.*

| # | Lesson | Goal |
|---|--------|------|
| 045 | Data that doesn't fit in memory | Process a file larger than RAM with chunking or generators, and keep the result reproducible. |
| 046 | Streaming, windows and late events | Compute a running aggregate over a stream and handle events that arrive out of order. |
| 047 | Training at a scale you can still debug | Shard a dataset, train in stages, and checkpoint so a crash doesn't throw the work away. |
| 048 | Capstone, part 1: choose and scope | Pick an advanced project, write a measurable plan, and gather or generate the data. |
| 049 | Capstone, part 2: build and evaluate | Build the pipeline, beat a simple baseline, and report uncertainty and failure cases. |
| 050 | Capstone, part 3: ship, reflect and what you'd do next | Package a minimal version, write it up, and leave a clear list of what you'd improve with more time. |

---

## After the curriculum

Once you're through 050, you should be able to read a paper or a library's
documentation in any of these areas and teach yourself the next thing. The
capstone is the place to specialise; the lessons before it are the shared
foundation.
