# Lesson 010 - Mini-project: decide with uncertainty

**Phase 1 - Bayesian thinking, causality and experiments** | Week 1, Day 5 | Thursday 2026-11-19

> **Goal:** Analyse a simulated experiment both Bayesian-style and
> frequentist-style, and write a recommendation a product lead could
> use.

Time: about 75 minutes to read and run, plus the write-up. Nothing new
to install.

---

This is the end of phase 1, and the point of the whole phase fits in
one sentence from its goal: go from "the model predicted X" to "here's
how sure we are, and here's what we should do about it". Today you do
that from start to finish.

The brief, from the head of product:

> We're thinking of offering every rider a **KES 200 weekend pass**:
> free weekend rides for 28 days. Launching it properly (app work,
> marketing, support training) costs about **KES 6 million**. Does it
> make us money? I need a yes or no, how sure you are, and what would
> change your mind.

We ran a randomised test: each rider got the offer or didn't, by coin
flip. Everything you need is in lessons 001-009. Nothing here is new
machinery; the new part is putting it together and **ending with a
decision**, not a p-value.

## How to follow along

Venv active, `python` in the repo root:

```bash
python lessons/phase-1-bayesian-and-causal/lesson-010-mini-project-decide-with-uncertainty/lesson.py
```

It runs in a couple of seconds and saves `output/decision.png`: the
posterior with the break-even line on the left, city by city on the
right.

Read each section's output and **say what you'd conclude before reading
my take**. That's the skill being practised.

## 0. Write the plan before you look

```
  primary metric                                 -> 28-day revenue per rider (KES), everyone assigned
  primary analysis                               -> difference in means, adjusted for pre-period rides
  riders per arm                                 -> 10,000
  one-off launch cost (KES)                      -> 6,000,000
  active riders per month                        -> 150,000
  horizon (months)                               -> 12
  break-even gain per rider-month                -> KES 3.33
```

Three choices here matter more than any formula later:

- **Revenue per rider, everyone assigned.** Not "revenue from pass
  buyers", and not take-up. The question is what happens to the
  business if we show everyone the offer, so we compare everyone who
  was offered with everyone who wasn't (the "intention to treat").
- **The adjustment is decided now.** Lesson 007 showed that adjusting
  for a pre-experiment covariate tightens the interval. Choosing to do
  it *after* seeing which version looks better is a quiet form of
  peeking.
- **The break-even line.** KES 6m over 150,000 riders for 12 months
  means the pass must add **KES 3.33 per rider per month** just to pay
  for itself. "Is the effect bigger than zero?" is the wrong question;
  "is it bigger than 3.33?" is the right one.

## 1. Check the coin

```
  riders: offer / control                        -> 10,148 / 9,989
  sample ratio mismatch p-value                  -> 0.26
  pre-period rides: offer / control              -> 5.12 / 5.00
  SMD of pre-period rides                        -> +0.025
```

The split isn't exactly 50/50, but a **sample
ratio mismatch** test (a chi-square test of the counts against the
planned split) says that's well within chance. If that p-value were
tiny, say below 0.001, you'd stop here: something in the assignment or
the logging is broken, and no analysis can repair it. Balance on last
month's rides is excellent (SMD well under 0.1). The coin did its job.

## 2. Frequentist: estimate, interval, p-value

```
  revenue per rider: offer / control             -> KES 311.3 / 299.8
  plain difference, 95% CI                       -> +11.47  [+3.83, +19.10]
  plain p-value                                  -> 0.003
  correlation of revenue with pre-period rides   -> 0.76
  adjusted difference, 95% CI                    -> +6.23  [+1.30, +11.16]
  adjusted p-value                               -> 0.013
  true effect (hidden in real life)              -> +4.68
  pass take-up in offer arm                      -> 12.9%
  rides per rider, difference                    -> +0.36  (SE 0.07)
```

The offer arm made KES 11.47 more per rider. But that arm also happened
to get slightly busier riders (5.12 rides last month vs 5.00), and with
a correlation of 0.76 between last month's rides and this month's
revenue, a little luck in who landed where moves the answer a lot. The
adjusted estimate, our **pre-registered primary analysis**, is **+6.23,
95% CI [+1.30, +11.16]**, with an interval about a third narrower.
(Because this is a simulation we can peek at the truth, +4.68. Both
intervals contain it; the adjusted estimate is much closer. Had we
reported the plain +11.47, we'd have nearly doubled the effect.)

13% of offered riders bought a pass, and riders as a whole took 0.36
more rides each. That's a nice side effect: more rides is the company's
mission as well as its revenue.

So the frequentist verdict is "significant at 5%". Notice what that
does **not** answer: whether it beats KES 3.33. The interval runs from
1.30, which loses money, to 11.16, which makes a lot.

## 3. A comparison to refuse

Someone will look at the data and say this:

```
  revenue: pass buyers / offered non-buyers      -> KES 486 / 285
  pre-period rides: buyers / non-buyers          -> 7.4 / 4.8
  'effect of buying', buyers minus control       -> +186
```

"Pass buyers spend KES 186 more than control riders! The pass is a
gold mine!" No. Buying the pass is something riders **chose**, after
the coin, and heavy riders chose it far more often (7.4 rides last
month vs 4.8). That's lesson 006's selection bias and lesson 008's
backdoor: `heavy rider -> buys pass` and `heavy rider -> revenue`. The
coin randomised the *offer*, not the *purchase*, so only comparisons
by offer are protected. Keep this one out of the slide deck.

## 4. Bayesian: how sure, and sure of what?

Now the questions the product lead actually asked. Since we have 10,000
riders per arm, the likelihood for the mean difference is very close to
normal, so the posterior is lesson 003's conjugate update with a normal
prior. Two priors:

- **Flat**: let the data speak alone.
- **Sceptical**, Normal(0, 8²): past pricing tests rarely moved revenue
  per rider by more than about KES 16 either way, so big effects need
  more evidence.

And a check that assumes no normality at all, the **Bayesian
bootstrap** (lesson 001's bootstrap, with random Dirichlet weights per
rider instead of resampling):

```
  Bayesian bootstrap: mean, sd                   -> +6.07, 2.53
  flat: mean, 95% interval                       -> +6.23  [+1.32, +11.15]
  flat: P(effect > 0)                            -> 0.993
  flat: P(effect > break-even)                   -> 0.877
  sceptical: mean, 95% interval                  -> +5.67  [+0.98, +10.36]
  sceptical: P(effect > 0)                       -> 0.991
  sceptical: P(effect > break-even)              -> 0.836
  take-up posterior Beta(a, b)                   -> Beta(1,318, 8,870)
  take-up 95% interval                           -> [0.123, 0.136]
```

The bootstrap agrees with the normal approximation, so we can trust it.
The sceptical prior pulls the estimate down a little, from 6.23 to
5.67; with this much data the prior can't do much. We use the sceptical
one from here on, because it's the more conservative of two defensible
choices.

The headline: it's **99% likely the pass raises revenue, and about 84%
likely it raises it enough to pay for itself**. That second number is
the one that matters, and no p-value gives it to you.

Take-up uses lesson 003's Beta-binomial, starting from past offers
(about 10%, worth 40 riders of evidence): between 12.3% and 13.6%.
Useful for the support team's planning.

## 5. City by city

The Nairobi operations lead will ask whether it works in Nairobi.

```
  overall effect across cities, mu               -> +7.79
  spread between cities, tau (posterior mean)    -> 5.20
    Nairobi  raw (SE) -> pooled [true]           ->  +1.74 ( 4.0) ->  +4.55  [+2.53]
    Mombasa  raw (SE) -> pooled [true]           ->  +8.56 ( 5.2) ->  +7.87  [+7.15]
    Kisumu   raw (SE) -> pooled [true]           ->  +7.25 ( 7.1) ->  +7.41  [+4.99]
    Nakuru   raw (SE) -> pooled [true]           ->  +8.62 ( 7.5) ->  +7.83  [+6.21]
    Eldoret  raw (SE) -> pooled [true]           ->  +9.15 ( 8.3) ->  +7.96  [+8.07]
    Thika    raw (SE) -> pooled [true]           -> +25.63 (10.9) -> +11.16  [+4.95]
  error vs truth (RMSE): raw / pooled            -> 8.59 / 2.93
```

Taken alone, Nairobi looks weak (+1.74) and Thika looks spectacular
(+25.63). Thika's is mostly noise: its standard error is 11. Lesson
005's hierarchical model estimates how much cities really differ, then
pulls each city towards the overall effect in proportion to how noisy
it is. As in lesson 005, we average over every plausible spread `tau`
rather than betting on the single best one (try the single best one in
exercise 5 and see what goes wrong). Thika's 25.6 shrinks to 11.2.
Nairobi, which has plenty of data, moves less, to 4.6.

Because this is a simulation, the last column shows the truth, and the
pooled estimates are far closer overall: a typical error of 2.9 against
8.6 for the raw ones. The truth also says something interesting:
Nairobi's effect really **is** the smallest (+2.53, below break-even).
Nairobi riders ride more, and the heavier the rider, the more fares a
pass replaces. So the city view is worth keeping an eye on, even though
no single city's raw number should be trusted.

One caveat about `mu`: the overall +7.79 here is above the per-rider
+6.23 from section 2, because partial pooling gives small cities more
say than they'd get by rider count. For the business decision, the
per-rider number is the right one; the city view is for spotting where
things might differ.

**Don't** ship city by city based on the raw numbers. That's six noisy
tests and lesson 007's winner's curse all at once.

## 6. From posterior to decision

Turn the posterior into money: gain per rider x 150,000 riders x 12
months, minus the KES 6m launch cost. Every posterior draw gives a
possible future; average over them.

```
  expected value of shipping (KES m)             -> +4.2
  P(shipping loses money)                        -> 0.164
  expected loss if we ship (KES m)               -> 0.37
  expected loss if we hold (KES m)               -> 4.59
  decision                                       -> ship
  most that more data could be worth (KES m)     -> 0.37
  expected gain given up per month of waiting    -> 0.85
  one-sided p, effect <= break-even              -> 0.124
```

Read it as a set:

- Shipping is worth about **KES 4.2m** in expectation, with a **16%**
  chance it loses money.
- **Expected loss** is the average amount you'd regret. If we ship and
  the effect turns out small, we'd lose on average KES 0.37m; if we
  hold and it turns out large, we'd miss on average KES 4.59m. The
  regrets are lopsided, so ship.
- Should we test longer instead? Even *perfect* information could only
  save us the expected loss of our best action, KES 0.37m (this is the
  **expected value of perfect information**). Another month of testing
  costs about KES 0.85m of expected gain. Stop testing.
- The frequentist test against break-even gives p = 0.124: we **can't
  reject** "it doesn't pay back" at 5%. That's not a contradiction. A
  test controls how often you're fooled into acting; a decision weighs
  how much each mistake costs. Here being wrong one way costs about
  twelve times more than the other, and the test doesn't know that.

## 7. The recommendation

This is the deliverable. It fits on one screen, it leads with the
answer, and it says what would change our mind.

> **Weekend pass: recommend launching to all riders.**
>
> **What we tested.** 20,137 riders, half shown the KES 200 weekend
> pass for 28 days, chosen by coin flip. Groups were balanced.
>
> **What happened.** Revenue rose by about **KES 6 per rider per
> month** (95% interval KES 1 to 11), and riders took about 0.4 more
> rides each. 13% of offered riders bought the pass.
>
> **Is it worth KES 6m to launch?** The pass needs KES 3.33 per rider
> per month to pay back within a year. We estimate an **84% chance it
> does**, with an expected 12-month gain of about **KES 4.2m** after
> launch costs. If we launch and we're wrong, the expected shortfall is
> small (about KES 0.4m); if we don't launch and we're wrong, we'd miss
> about KES 4.6m. More testing isn't worth the delay.
>
> **What to watch.** Nairobi showed the weakest result of the six
> cities, and heavy riders (common in Nairobi) gain least from a pass.
> Track Nairobi revenue per rider against a small holdout group for the
> first two months; if it isn't clearly above KES 3 per rider, revisit
> the price there. Don't judge the pass on how much pass buyers spend;
> they were heavier riders to begin with.
>
> **What would change our mind.** A launch cost above about KES 10m,
> or evidence the effect fades after the first month (we only measured
> 28 days).

Notice what's *not* in it: no "p = 0.013", no "statistically
significant", no Beta distributions. Those live in an appendix for
anyone who asks. The product lead gets an action, a probability, the
stakes in shillings, and a tripwire.

## What phase 1 gave you

- **001-002:** uncertainty is the answer, not a footnote; beliefs as
  distributions, updated by data.
- **003-005:** priors you can compute on paper, sampling when you
  can't, and partial pooling for many small groups (section 5 above).
- **006-008:** the causal question is a comparison of what would have
  happened; randomise when you can; draw the graph when you can't
  (section 3 above).
- **009:** when you can't randomise *or* measure the confounder, find a
  design like difference-in-differences.
- **010:** put it together and end with a decision, priced in the
  units your audience cares about.

## What to do now

1. Run `lesson.py` and compare both panels of `output/decision.png`
   with the memo.
2. Do the exercises in [`exercises.md`](exercises.md). The last one is
   writing your own memo for a different result.
3. Phase 2 starts tomorrow with lesson 011: time series as a modelling
   problem, where order matters and a random split will lie to you.
