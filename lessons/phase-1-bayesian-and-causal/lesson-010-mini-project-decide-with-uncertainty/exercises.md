# Lesson 010 - Exercises

Two quick checks, three hands-on experiments, and the real deliverable:
a memo of your own. Predict first, then run.

Activate your venv. Copy `lesson.py` somewhere of your own and edit the
copy. From the repo root:

```bash
mkdir -p my_work_010
cp lessons/phase-1-bayesian-and-causal/lesson-010-mini-project-decide-with-uncertainty/lesson.py my_work_010/
cd my_work_010
python lesson.py
```

---

## 1. Say it right

The product lead reads four sentences in a draft. Which are correct
readings of this lesson's results, and what's wrong with the others?

a) "There's a 1.3% chance the pass has no effect." (adjusted p = 0.013)
b) "There's about an 84% chance the pass pays back its launch cost
within a year."
c) "The result is significant, so the pass will make money."
d) "We're 95% sure the effect is between KES 1 and KES 11 per rider."

<details>
<summary>Check yourself</summary>

a) **Wrong.** A p-value is the chance of data this extreme *if* there
were no effect, not the chance there's no effect. Lesson 002's base-rate
trap in a new outfit.

b) **Right**, given the model and the sceptical prior. That's the
posterior probability from section 4, and it's the sentence a decision
maker can use.

c) **Wrong** twice. Significant means "probably not zero", but the pass
needs more than zero; it needs KES 3.33. And even then it's a
probability, not a promise.

d) **Fine in practice.** Strictly, a frequentist 95% CI is a statement
about the procedure, but here the flat-prior Bayesian interval is
almost identical (+1.32 to +11.15), so this reading is fair. With small
samples or strong priors the two can diverge, and then the wording
matters.

</details>

## 2. Stop or go?

Before running anything: what would you do in each case?

a) The sample ratio mismatch p-value is 0.0001.
b) Halfway through the test, the head of marketing asks you to look at
the numbers "just to see", and they look great.
c) After the test, someone notices the effect is biggest for riders who
used the app on Sundays and suggests launching only to them.

<details>
<summary>Check yourself</summary>

a) **Stop.** The split is broken (a bug in assignment, or one arm's
events getting lost). Find the cause; don't analyse the results.

b) **Look if you must, but don't decide.** Lesson 007's peeking result:
stopping when it looks good inflates false positives. If early stopping
is a real need, plan it in advance with a sequential design.

c) **Treat it as a new hypothesis, not a finding.** It was found by
searching many subgroups (lesson 007's winner's curse again). Test it
in a fresh experiment if it's worth it.

</details>

## 3. Hands-on: what if launch costs double?

In your copy, change the launch cost in `PLAN` to `12_000_000`. Predict
first: ship or hold? And is more testing worth it now?

<details>
<summary>Check yourself</summary>

```
  break-even gain per rider-month                -> KES 6.67
  sceptical: P(effect > break-even)              -> 0.340
  expected value of shipping (KES m)             -> -1.8
  expected loss if we ship (KES m)               -> 2.75
  expected loss if we hold (KES m)               -> 0.97
  decision                                       -> hold
  most that more data could be worth (KES m)     -> 0.97
  expected gain given up per month of waiting    -> 0.85
```

Break-even doubles to 6.67, right on top of our estimate, so the call
flips to **hold**. And now the value of more information (0.97m) is a
little *above* the cost of a month's delay (0.85m), so the sensible
next step is not "no" but "run it longer". Same data, different
stakes, different decision. That's the whole point of section 6.

</details>

## 4. Hands-on: a much more sceptical prior

Put the cost back. Change the sceptical prior from `8.0` to `3.0`
(you'd do this if past pricing tests almost never moved revenue by more
than about KES 6 a rider).

<details>
<summary>Check yourself</summary>

```
  sceptical: mean, 95% interval                  -> +3.66  [-0.11, +7.43]
  sceptical: P(effect > break-even)              -> 0.568
  expected value of shipping (KES m)             -> +0.6
  decision                                       -> ship
  most that more data could be worth (KES m)     -> 1.10
  expected gain given up per month of waiting    -> 0.55
```

With a strong prior, 10,000 riders per arm can't overrule it: the
estimate is dragged from 6.23 to 3.66. Shipping still edges it, but
it's close, and now more data is worth twice what a month's delay
costs. So the honest recommendation becomes "extend the test". This is
why a memo should say which prior you used and why, and ideally show
that the decision survives a reasonable alternative (ours did, with
the original 8.0 and the flat prior).

</details>

## 5. Hands-on: bet on one tau

Section 5 averages over every plausible `tau`. Replace the line
`shrunk = wt @ np.array(cond)` with:

```python
k = int(np.argmax(ll))
shrunk = np.array(cond)[k]
print("  best single tau", taus[k])
```

What happens to the cities, to the RMSE, and to Nairobi in particular?

<details>
<summary>Check yourself</summary>

```
  best single tau 1.25
    Nairobi  raw (SE) -> pooled [true]           ->  +1.74 ( 4.0) ->  +6.37  [+2.53]
    ...
    Thika    raw (SE) -> pooled [true]           -> +25.63 (10.9) ->  +7.06  [+4.95]
  error vs truth (RMSE): raw / pooled            -> 8.59 / 2.02
```

The single best `tau` is small, so every city is squeezed to about 6.4
to 7.1. The RMSE even looks better (2.02 vs 2.93), because most cities
really are similar. But look at Nairobi: it lands at +6.37, comfortably
above break-even, while the truth is +2.53, below it. Betting on one
`tau` claimed more certainty than six cities can give, and erased the
one city that needed watching. Averaging over `tau` kept Nairobi at
4.55, still flagged as the weakest. Good on average and right where it
matters are different goals.

</details>

## 6. The deliverable: write the other memo

Put everything back to the original, then change the seed at the top
to `np.random.default_rng(17)`. This is the same world (same true
effect) with a different set of riders. Run it, then write a memo of
at most 250 words for the head of product, using section 7 of the
README as the template: recommendation first, what we tested, what
happened, is it worth it, what to watch, what would change your mind.

<details>
<summary>Check yourself</summary>

The key numbers:

```
  adjusted difference, 95% CI                    -> +2.28  [-2.66, +7.22]
  true effect (hidden in real life)              -> +4.70
  sceptical: P(effect > break-even)              -> 0.301
  expected value of shipping (KES m)             -> -2.3
  decision                                       -> hold
  most that more data could be worth (KES m)     -> 0.83
  expected gain given up per month of waiting    -> 0.31
```

A good memo says something like: **don't launch yet; extend the test
for a month.** The effect is probably positive (about 81%) but probably
too small to pay back KES 6m (30% chance it does). Information is worth
more than the delay costs (0.83m vs 0.31m), so buying more data is the
best action, not a cop-out.

Then the uncomfortable part: the true effect here is +4.70, above
break-even. This sample was unlucky, and holding would be a mistake in
hindsight. That's fine. **A good decision is one that's right given what
you knew, not one that happens to turn out well.** The memo that
recommends more data protects you from exactly this kind of bad luck.
Also notice the cities: Nairobi raw is -6.86, which is noise (truth
+2.57). A memo that said "it hurts Nairobi" would be overreading.

</details>
