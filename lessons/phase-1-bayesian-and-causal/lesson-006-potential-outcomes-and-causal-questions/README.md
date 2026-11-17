# Lesson 006 - Potential outcomes and causal questions

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 3 | Tuesday 2026-11-17

> **Goal:** Write a decision as a contrast of potential outcomes and see
> why a good predictor can still be a bad policy.

Time: about 45 minutes to read and run, plus the exercises. Nothing new
to install.

---

Everything in the first course, and in this track so far, answered
questions like "how likely is this?" Today's question is different:
**"what happens if we do something?"**

The operations team wants to text a reminder to riders the evening
before a return is due. They have budget for 4,000 texts. The tempting
plan is: build a model of who is likely to be late, text the riskiest
4,000. That sounds like ML doing its job. It can be a quiet disaster,
and the reason is the single most important idea in causal inference.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-006-potential-outcomes-and-causal-questions/lesson.py
```

It runs in a second or two and saves `output/uplift.png` next to the
script.

## 1. Two outcomes per rider

For each rider, imagine two versions of tomorrow:

- `Y0`: are they late **if we don't** text them?
- `Y1`: are they late **if we do**?

These are **potential outcomes**. The effect of the text on one rider is
`Y1 - Y0`: -1 if the text saves a late return, 0 if it changes nothing.
The **average treatment effect** (ATE) is the average of that over
everyone.

In a simulation we can play god and generate both. We give 20,000
riders two features (late rides out of their last 10, and trip
distance) and decide that the text helps forgetful occasional riders
most. Riders who are late nearly every time, or who ride a long way,
are late for reasons a text can't fix.

```
  share late without / with reminder             -> 0.223 / 0.173
  true ATE  E[Y1 - Y0]                           -> -0.050
  riders the text 'saves' (Y0=1, Y1=0)           -> 1,000 of 20,000
```

Texting everyone would cut late returns by 5 percentage points. Keep
that number in mind; nothing in the real data will hand it to us.

## 2. We only ever see one

Here's the catch. Each real rider was either texted or not. We see `Y1`
for the texted and `Y0` for the rest, never both:

```
observed Y = Y1 if texted else Y0
```

This is called the **fundamental problem of causal inference**. The
other outcome, the **counterfactual**, is missing for every single
person, so no amount of data lets us compute `Y1 - Y0` for an
individual. Causal inference is the craft of estimating averages of it
anyway.

Last month's texts weren't random: operations texted riders they judged
risky, so the more past late rides you had, the likelier you were to get
one.

```
  riders texted                                  -> 29.9%
```

## 3. The naive comparison

The obvious analysis compares late rates of texted and untexted riders:

```
  late | texted  minus  late | not texted        -> +0.082
```

Taken at face value: texts make people 8 points *more* likely to be
late. Cancel the programme! But we know the true effect is negative.
Algebra splits the naive difference into two pieces:

```
    = ATT  E[Y1 - Y0 | texted]                   -> -0.052
    + selection bias  E[Y0|T=1] - E[Y0|T=0]      -> +0.134
  check: ATT + bias                              -> +0.082
```

- **ATT**, the average effect on the treated, is what the texts actually
  did for the riders who got them: a 5-point drop.
- **Selection bias** is how different the two groups would have been
  *anyway*, with no texts at all. Texted riders were chosen for being
  risky, so they'd have been 13 points later even untouched.

The bias swamps the effect and flips its sign. Whenever who gets
treated depends on things that also drive the outcome, the naive
comparison measures a mix of both.

Now flip a coin for each rider instead:

```
  randomised: late | texted minus not            -> -0.049   (true ATE -0.050)
```

Randomisation makes `T` unrelated to `Y0`, so the selection-bias term is
zero on average and the simple difference recovers the true effect.
That's why randomised experiments are the gold standard, and lesson 007
is all about running them well.

## 4. A good predictor

Now the model. A logistic regression of lateness on the two features,
fitted to riders who weren't texted (so it predicts `Y0`):

```
  coefficients (intercept, past_late, dist)      -> [-2.58, 0.42, 0.15]
  AUC for 'late without a text' (all riders)     -> 0.721
```

It recovers the true coefficients we simulated (-2.6, 0.45, 0.15) and
ranks riders by risk respectably. As a predictor, it's done its job.

## 5. Two policies, same budget

We have 4,000 texts. Compare three ways to spend them:

```
  late returns, nobody texted                    -> 4,464
    random 4,000: late returns prevented         -> 198
    top 4,000 by risk: late returns prevented    -> 134
    top 4,000 by uplift: late returns prevented  -> 413
  mean past_late: risk pick / uplift pick        -> 4.3 / 2.4
```

The risk-based policy, the one built on a good model, prevents **fewer**
late returns than texting at random. Targeting by **uplift**, the
expected reduction `P(Y0=1) - P(Y1=1)`, prevents about three times as
many. (We can only compute true uplift because this is a simulation;
estimating it from data is what experiments and causal models are for.)

The reason is in the last line: the riskiest riders are the chronic
late-returners the text can't help. The model was answering "who will
be late?" when the decision needed "**who will be late unless we act,
and on time if we do?**"

## 6. Who does the text actually help?

Split riders into five bands by predicted risk:

```
    risk quintile 1: late w/o text, true uplift  -> 0.090, +0.019
    risk quintile 2: late w/o text, true uplift  -> 0.129, +0.045
    risk quintile 3: late w/o text, true uplift  -> 0.166, +0.069
    risk quintile 4: late w/o text, true uplift  -> 0.253, +0.084
    risk quintile 5: late w/o text, true uplift  -> 0.478, +0.034
```

Risk climbs steadily; uplift rises and then collapses in the top band.
Open `output/uplift.png`. On the left, each dot is a rider: predicted
risk across, true uplift up. The highest-uplift riders sit in the
middle of the risk range. On the right, the cumulative late returns
prevented as you send more texts: the uplift ordering pulls ahead
immediately, and the two only meet when you text everyone.

People who work on this name four kinds of customer: **persuadables**
(helped by the action), **sure things** (fine either way), **lost
causes** (late either way), and **sleeping dogs** (made worse by the
action). A risk model can't tell persuadables from lost causes; both
look risky.

## The general lesson

Prediction asks about `Y`. Decisions ask about `Y1 - Y0`. Whenever a
model's output will be used to choose an action (who gets a discount,
which patients get a follow-up call, which machines get serviced, which
loans get a hardship plan) ask out loud: **am I predicting the outcome,
or the effect of my action on it?** They often point in different
directions.

## What to remember

- Every unit has **potential outcomes** `Y0` and `Y1`; the effect is
  their difference, and we only ever observe **one** of them.
- **ATE** averages the effect over everyone; **ATT** over those who got
  the treatment.
- A naive treated-vs-untreated comparison equals **ATT plus selection
  bias**, and the bias can flip the sign.
- **Randomisation** removes selection bias on average.
- **High risk is not high uplift.** Targeting by predicted outcome can
  be worse than random when the action helps the middle, not the top.

## What to do now

1. Run `lesson.py` and look at both panels of `output/uplift.png`.
2. Do the exercises in [`exercises.md`](exercises.md).
3. Lesson 007 designs the experiment that would have told operations
   the truth: a proper A/B test with a confidence interval.
