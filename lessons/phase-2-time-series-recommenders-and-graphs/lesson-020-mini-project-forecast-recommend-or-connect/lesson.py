"""
Lesson 020 - Mini-project: forecast, recommend or connect

The end of phase 2. Pick one of three threads and take it all the way to a
short, evaluated write-up. This script works one track in full, on a small
public dataset, and gives you a running start on the other two.

  Track A (worked here): FORECAST next week's daily rentals for a real
  bike-share system (Capital Bikeshare, Washington DC, 2011-2012; the UCI
  "Bike Sharing" dataset, saved next to this script as day.csv).

  Track B (starter): RECOMMEND the next stations a rider will use.
  Track C (starter): CONNECT, on Zachary's karate club, a classic public
  social network: predict who sides with whom, and which ties exist.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/forecast.png`, next to
this script (ignored by git), and runs in under a minute on a laptop.

Data: Fanaee-T, H. and Gama, J. (2013), "Event labeling combining ensemble
detectors and background knowledge", Progress in Artificial Intelligence,
doi:10.1007/s13748-013-0040-3. UCI Machine Learning Repository, CC BY 4.0.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
H = 7  # forecast horizon, days


def show(label, value):
    print(f"  {label:<50} -> {value}")


rows = list(csv.DictReader(open(HERE / "day.csv")))
dates = np.array([r["dteday"] for r in rows])
cnt = np.array([float(r["cnt"]) for r in rows])
col = lambda k: np.array([float(r[k]) for r in rows])
weekday, workday, holiday = col("weekday").astype(int), col("workingday"), col("holiday")
weather = np.column_stack([col("weathersit") == 2, col("weathersit") >= 3, col("temp"), col("atemp"), col("hum"), col("windspeed")]).astype(float)
doy = np.array([(np.datetime64(d) - np.datetime64(d[:4] + "-01-01")).astype(int) for d in dates])
N = len(cnt)
VAL_ORIGINS = np.arange(int(np.where(dates == "2012-03-04")[0][0]), int(np.where(dates == "2012-11-04")[0][0]), 7)
TEST_ORIGINS = np.arange(int(np.where(dates == "2012-11-04")[0][0]), N - H, 7)

# ---------------------------------------------------------------------------
# 0. The plan, written before looking at any results
# ---------------------------------------------------------------------------
print("\n0. The plan")
show("question", "total rentals on each of the next 7 days")
show("who uses it", "ops lead: van shifts and maintenance days")
show("metric", "MAE in rentals/day, plus 80% interval coverage")
show("forecast made", "every Sunday night, for Mon-Sun")
show("model selection on (weekly origins)", f"{dates[VAL_ORIGINS[0] + 1]} to {dates[VAL_ORIGINS[-1] + H]}  ({len(VAL_ORIGINS)} weeks)")
show("sealed test, used once (weekly origins)", f"{dates[TEST_ORIGINS[0] + 1]} to {dates[TEST_ORIGINS[-1] + H]}  ({len(TEST_ORIGINS)} weeks)")
show("beat this or don't ship", "seasonal naive (same weekday last week)")

# ---------------------------------------------------------------------------
# 1. Look at the data (training part only)
# ---------------------------------------------------------------------------
print("\n1. The data")
show("days / first / last", f"{N} / {dates[0]} / {dates[-1]}")
show("mean rentals per day: 2011 / 2012", f"{cnt[:365].mean():.0f} / {cnt[365:].mean():.0f}  (+{cnt[365:].mean() / cnt[:365].mean() - 1:.0%})")
tr = slice(0, VAL_ORIGINS[0] + 1)
show("before 2012-03: mean by weekday (Sun..Sat)", " ".join(f"{cnt[tr][weekday[tr] == k].mean():.0f}" for k in range(7)))
show("before 2012-03: mean on clear / misty / wet days", " / ".join(f"{m:.0f}" for m in [cnt[tr][col('weathersit')[tr] == 1].mean(), cnt[tr][col('weathersit')[tr] == 2].mean(), cnt[tr][col('weathersit')[tr] >= 3].mean()]))
lag7 = np.corrcoef(cnt[tr][7:], cnt[tr][:-7])[0, 1]
lag1 = np.corrcoef(cnt[tr][1:], cnt[tr][:-1])[0, 1]
show("before 2012-03: autocorrelation at lag 1 / lag 7", f"{lag1:.2f} / {lag7:.2f}")


# ---------------------------------------------------------------------------
# Forecasters. Each takes an origin t (last day we know) and returns 7 numbers
# for days t+1..t+7, using only data up to and including t.
# ---------------------------------------------------------------------------
def naive(t):
    return np.full(H, cnt[t])


def seasonal_naive(t):
    return np.array([cnt[t + h - 7] for h in range(1, H + 1)])


def level(t):
    return cnt[t - 6:t + 1].mean()


def features(s, h, wx=None):
    """Row describing 'forecast day s+h from origin s'. Everything about the
    target day here is known in advance (calendar), except the weather."""
    j = s + h
    lv = level(s)
    f = [np.log(cnt[j - 7] / lv), np.log(cnt[s] / lv), np.log(lv / cnt[s - 27:s - 6].mean())]
    f += list(np.eye(H)[h - 1]) + list(np.eye(7)[weekday[j]]) + [workday[j], holiday[j]]
    f += [np.sin(2 * np.pi * k * doy[j] / 365.25) for k in (1, 2)] + [np.cos(2 * np.pi * k * doy[j] / 365.25) for k in (1, 2)]
    if wx is not None:
        f += list(wx)
    return f


def weather_forecast(j, h, rng):
    """What we'd actually know on Sunday about day j, h days ahead: the true
    weather blurred more the further out it is (temperature off by ~1.5 C per
    sqrt(day), and the rain call wrong more often), like a real forecast."""
    w = weather[j].copy()
    w[2:4] += rng.normal(0, 0.035 * np.sqrt(h), 2)
    w[4:6] += rng.normal(0, 0.05 * np.sqrt(h), 2)
    if rng.random() < 0.04 * h:
        w[:2] = np.eye(3)[rng.integers(0, 3)][1:]
    return w


def make_ml(kind="ridge", wx="none", seed=0):
    def fc(t):
        rng = np.random.default_rng(seed + t)
        Xtr, ytr = [], []
        for s in range(35, t):
            for h in range(1, H + 1):
                if s + h > t:
                    break  # the target must already be known at time t
                w = None if wx == "none" else (weather[s + h] if wx == "oracle" else weather_forecast(s + h, h, rng))
                Xtr.append(features(s, h, w))
                ytr.append(np.log(cnt[s + h] / level(s)))
        Xtr, ytr = np.array(Xtr), np.array(ytr)
        if kind == "ridge":
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            m = Ridge(alpha=3.0).fit((Xtr - mu) / sd, ytr)
            pred = lambda X: m.predict((X - mu) / sd)
        else:
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
            pred = m.predict
        Xf = []
        for h in range(1, H + 1):
            w = None if wx == "none" else (weather[t + h] if wx == "oracle" else weather_forecast(t + h, h, rng))
            Xf.append(features(t, h, w))
        return level(t) * np.exp(pred(np.array(Xf)))
    return fc


def backtest(fc, origins):
    P = np.array([fc(t) for t in origins])
    Y = np.array([cnt[t + 1:t + H + 1] for t in origins])
    return P, Y


models = {
    "naive (last day)": naive,
    "seasonal naive": seasonal_naive,
    "ridge: calendar + lags": make_ml("ridge"),
    "ridge: + forecast weather": make_ml("ridge", "forecast"),
    "boosting: + forecast weather": make_ml("hgb", "forecast"),
    "ridge: + PERFECT weather (not allowed)": make_ml("ridge", "oracle"),
}

# ---------------------------------------------------------------------------
# 2. Baselines and candidates on the selection weeks
# ---------------------------------------------------------------------------
print("\n2. Rolling-origin backtest on the selection weeks (MAE, rentals/day)")
val = {k: backtest(f, VAL_ORIGINS) for k, f in models.items()}
mae_sn = np.abs(val["seasonal naive"][0] - val["seasonal naive"][1]).mean()
for k, (P, Y) in val.items():
    m = np.abs(P - Y).mean()
    show(k, f"{m:.0f}   skill vs seasonal naive {1 - m / mae_sn:+.0%}")
honest = [k for k in models if "PERFECT" not in k]
best = min(honest, key=lambda k: np.abs(val[k][0] - val[k][1]).mean())
show("chosen (best honest model on selection weeks)", best)

# ---------------------------------------------------------------------------
# 3. Is the gain real? Bootstrap over weeks
# ---------------------------------------------------------------------------
print("\n3. Is the gain real?")
P, Y = val[best]
Ps, _ = val["seasonal naive"]
gain_week = np.abs(Ps - Y).mean(1) - np.abs(P - Y).mean(1)
rng = np.random.default_rng(20)
boot = np.array([gain_week[rng.integers(0, len(gain_week), len(gain_week))].mean() for _ in range(5000)])
show("MAE gain per day vs seasonal naive, 95% CI", f"{gain_week.mean():.0f}  [{np.percentile(boot, 2.5):.0f}, {np.percentile(boot, 97.5):.0f}]")
show("weeks where the chosen model wins", f"{(gain_week > 0).sum()} of {len(gain_week)}")
show("MAE by horizon (day 1..7)", " ".join(f"{v:.0f}" for v in np.abs(P - Y).mean(0)))
err = (P - Y).ravel()
when = np.array([dates[t + 1:t + H + 1] for t in VAL_ORIGINS]).ravel()
for i in np.argsort(-np.abs(err))[:3]:
    show(f"big miss: {when[i]}", f"forecast {P.ravel()[i]:.0f}, actual {Y.ravel()[i]:.0f}")

# ---------------------------------------------------------------------------
# 4. Intervals from past errors
# ---------------------------------------------------------------------------
print("\n4. 80% intervals from the selection-week errors")
ratio = np.log(Y / P)  # log errors, per horizon
lo_q, hi_q = np.percentile(ratio, 10, axis=0), np.percentile(ratio, 90, axis=0)
show("interval width at day 1 / day 7 (x forecast)", f"{np.exp(lo_q[0]):.2f}-{np.exp(hi_q[0]):.2f} / {np.exp(lo_q[-1]):.2f}-{np.exp(hi_q[-1]):.2f}")
show("coverage on selection weeks (in-sample, ~80%)", f"{((ratio >= lo_q) & (ratio <= hi_q)).mean():.0%}")

# ---------------------------------------------------------------------------
# 5. Open the sealed test, once
# ---------------------------------------------------------------------------
print("\n5. Sealed test (November-December 2012), opened once")
test = {k: backtest(models[k], TEST_ORIGINS) for k in ["seasonal naive", best, "ridge: + PERFECT weather (not allowed)"]}
Pt, Yt = test[best]
mae_t_sn = np.abs(test["seasonal naive"][0] - Yt).mean()
for k, (Pk, _) in test.items():
    m = np.abs(Pk - Yt).mean()
    show(k, f"MAE {m:.0f}   skill {1 - m / mae_t_sn:+.0%}")
lo, hi = Pt * np.exp(lo_q), Pt * np.exp(hi_q)
cover = (Yt >= lo) & (Yt <= hi)
show("80% interval coverage on test", f"{cover.mean():.0%}  ({cover.sum()} of {cover.size} days)")
tw = np.array([dates[t + 1:t + H + 1] for t in TEST_ORIGINS])
xmas = np.isin(tw, [f"2012-12-{d}" for d in range(24, 32)])
show("test MAE: Christmas week vs the rest", f"{np.abs(Pt - Yt)[xmas].mean():.0f} vs {np.abs(Pt - Yt)[~xmas].mean():.0f}")
show("test bias (forecast - actual), mean", f"{(Pt - Yt).mean():+.0f} rentals/day")

# ---------------------------------------------------------------------------
# 6. The memo (fill it from the numbers above)
# ---------------------------------------------------------------------------
print("\n6. Memo")
print(f"""  Recommendation: use '{best}' for the weekly plan.
  On {len(TEST_ORIGINS)} unseen weeks it was off by {np.abs(Pt - Yt).mean():.0f} rentals a day on average,
  against {mae_t_sn:.0f} for 'same as last week' ({1 - np.abs(Pt - Yt).mean() / mae_t_sn:.0%} better). Each forecast comes with an
  80% range, but on test it held only {cover.mean():.0%} of the time ({cover[~xmas].mean():.0%} outside Christmas week):
  the ranges are too narrow for holidays and storms, which the model can't see coming.
  Plan those days by hand, and re-check the ranges after the holidays.""")

# ---------------------------------------------------------------------------
# 7. Starters for the other two tracks
# ---------------------------------------------------------------------------
print("\n7a. Track B starter: recommend the next stations (simulated riders)")
rng = np.random.default_rng(7)
R_, S_, DAYS = 400, 50, 60
pop = rng.dirichlet(np.full(S_, 0.6))
taste = rng.dirichlet(np.full(S_, 0.15), R_)  # each rider has a few favourite stations
trips = []
for u in range(R_):
    pref = 0.7 * taste[u] + 0.3 * pop
    k = rng.poisson(0.6 * DAYS)
    trips += [(u, s, d) for s, d in zip(rng.choice(S_, k, p=pref), rng.integers(0, DAYS, k))]
trips = np.array(trips)
train_t, test_t = trips[trips[:, 2] < 45], trips[trips[:, 2] >= 45]
M = np.zeros((R_, S_))
np.add.at(M, (train_t[:, 0], train_t[:, 1]), 1)
truth_sets = [set(test_t[test_t[:, 0] == u, 1]) for u in range(R_)]
users = [u for u in range(R_) if truth_sets[u] and M[u].sum() > 0]


def recall_at(scores, k=5):
    return np.mean([len(set(np.argsort(-scores[u])[:k]) & truth_sets[u]) / min(k, len(truth_sets[u])) for u in users])


Mn = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-9)
item_sim = Mn.T @ Mn
np.fill_diagonal(item_sim, 0)
show("riders / stations / train trips / test trips", f"{R_} / {S_} / {len(train_t)} / {len(test_t)}")
show("recall@5: most popular stations", f"{recall_at(np.tile(M.sum(0), (R_, 1))):.3f}")
show("recall@5: rider's own most-used stations", f"{recall_at(M + 1e-3 * M.sum(0)):.3f}")
show("recall@5: item-item similarity", f"{recall_at(M @ item_sim):.3f}")

print("\n7b. Track C starter: connect (Zachary's karate club, 34 members)")
KARATE = ("0-1 0-2 0-3 0-4 0-5 0-6 0-7 0-8 0-10 0-11 0-12 0-13 0-17 0-19 0-21 0-31 1-2 1-3 1-7 1-13 1-17 1-19 1-21 "
          "1-30 2-3 2-7 2-8 2-9 2-13 2-27 2-28 2-32 3-7 3-12 3-13 4-6 4-10 5-6 5-10 5-16 6-16 8-30 8-32 8-33 9-33 "
          "13-33 14-32 14-33 15-32 15-33 18-32 18-33 19-33 20-32 20-33 22-32 22-33 23-25 23-27 23-29 23-32 23-33 "
          "24-25 24-27 24-31 25-31 26-29 26-33 27-33 28-31 28-33 29-32 29-33 30-32 30-33 31-32 31-33 32-33")
EK = np.array([list(map(int, e.split("-"))) for e in KARATE.split()])
club = np.ones(34, int)
club[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]] = 0  # 0 = sided with the instructor (node 0)
AK = np.zeros((34, 34))
AK[EK[:, 0], EK[:, 1]] = AK[EK[:, 1], EK[:, 0]] = 1
dk = AK.sum(1)
SK = AK / np.sqrt(dk)[:, None] / np.sqrt(dk)[None]
Y0 = np.zeros((34, 2))
Y0[0, 0] = Y0[33, 1] = 1  # we only know the two leaders
F = 0.1 * np.linalg.solve(np.eye(34) - 0.9 * SK, Y0)
show("members / ties / homophily by final club", f"34 / {len(EK)} / {(club[EK[:, 0]] == club[EK[:, 1]]).mean():.0%}")
show("label propagation from the 2 leaders: accuracy", f"{(F.argmax(1) == club).mean():.0%}  (wrong: {np.where(F.argmax(1) != club)[0].tolist()})")
rng = np.random.default_rng(3)
hide = rng.choice(len(EK), 8, replace=False)
Atr = AK.copy()
Atr[EK[hide, 0], EK[hide, 1]] = Atr[EK[hide, 1], EK[hide, 0]] = 0
non = np.array([(a, b) for a in range(34) for b in range(a + 1, 34) if AK[a, b] == 0])
neg = non[rng.choice(len(non), 8, replace=False)]
pairs = np.vstack([EK[hide], neg])
truth = np.r_[np.ones(8), np.zeros(8)]
aa = (Atr[pairs[:, 0]] * Atr[pairs[:, 1]] / np.log(np.maximum(Atr.sum(1), 2))).sum(1)
show("link prediction, 8 hidden ties: Adamic-Adar AUC", f"{roc_auc_score(truth, aa):.2f}  (only 16 pairs: very noisy)")

# ---------------------------------------------------------------------------
# Figure: the sealed test weeks, forecast paths and intervals
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(TEST_ORIGINS[0] - 27, N)
ax[0].plot(x, cnt[x], color="k", lw=1, label="actual")
for i, t in enumerate(TEST_ORIGINS):
    xs = np.arange(t + 1, t + H + 1)
    ax[0].fill_between(xs, lo[i], hi[i], color="tab:blue", alpha=0.2, lw=0)
    ax[0].plot(xs, Pt[i], color="tab:blue", lw=1.5, label="forecast + 80% range" if i == 0 else None)
    ax[0].plot(xs, test["seasonal naive"][0][i], color="tab:orange", lw=1, ls="--", label="seasonal naive" if i == 0 else None)
ticks = x[::14]
ax[0].set_xticks(ticks)
ax[0].set_xticklabels([dates[i][5:] for i in ticks], fontsize=8)
ax[0].set_ylabel("rentals per day")
ax[0].set_title("Sealed test: weekly forecasts, Nov-Dec 2012")
ax[0].legend(fontsize=8, loc="lower left")
for k in ["seasonal naive", "ridge: calendar + lags", best]:
    Pk, Yk = val[k]
    ax[1].plot(range(1, H + 1), np.abs(Pk - Yk).mean(0), marker="o", label=k)
ax[1].set_xlabel("days ahead")
ax[1].set_ylabel("MAE (selection weeks)")
ax[1].set_title("Error by horizon")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "forecast.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/forecast.png")
