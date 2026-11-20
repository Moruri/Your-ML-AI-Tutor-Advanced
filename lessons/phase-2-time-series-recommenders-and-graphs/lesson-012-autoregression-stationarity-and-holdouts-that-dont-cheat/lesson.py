"""
Lesson 012 - Autoregression, stationarity and holdouts that don't cheat

Lesson 011 set the bar: the seasonal naive forecast, "same as last week".
Today we build the first model that uses the past properly, an
autoregression, and learn the two things that decide whether it can be
trusted: is the series stable enough to model (stationarity), and was it
tested the way it will be used (a rolling origin, never a random split).

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/rolling_origin.png`,
next to this script (ignored by git), and runs in about five seconds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def show(label, value):
    print(f"  {label:<46} -> {value}")


# Same series as lesson 011: same function, same seed.
HOLIDAYS = ["01-01", "05-01", "06-01", "10-20", "12-12", "12-25", "12-26"]


def rides_series(n_days=700, seed=11):
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2024-01-01") + np.arange(n_days)
    t = np.arange(n_days)
    dow = t % 7
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    trend = 900 + 0.55 * t
    weekly = np.array([60, 75, 80, 75, 70, -110, -250])[dow]
    rain = -120 * np.exp(-0.5 * ((doy - 120) / 22) ** 2) - 70 * np.exp(-0.5 * ((doy - 320) / 15) ** 2)
    md = np.array([str(d)[5:] for d in dates])
    holiday = np.where(np.isin(md, HOLIDAYS), -300, 0)
    noise = np.zeros(n_days)
    for i in range(n_days):
        noise[i] = (0.6 * noise[i - 1] if i else 0) + rng.normal(0, 35)
    return dates, np.round(trend + weekly + rain + holiday + noise)


dates, y = rides_series()
rng = np.random.default_rng(12)

# ---------------------------------------------------------------------------
# 1. What stationarity means, on three toy series
# ---------------------------------------------------------------------------
print("\n1. Stationary or not?")
N = 500
eps = rng.normal(0, 1, N)
ar1 = np.zeros(N)
for i in range(1, N):
    ar1[i] = 0.7 * ar1[i - 1] + eps[i]
walk = np.cumsum(eps)
for name, s in [("white noise", eps), ("AR(1), phi = 0.7", ar1), ("random walk", walk)]:
    a, b = s[:N // 2], s[N // 2:]
    show(f"{name}: mean, sd (1st | 2nd half)", f"{a.mean():+5.2f}, {a.std():4.2f} | {b.mean():+5.2f}, {b.std():4.2f}")

# Many random walks: how far do they wander by step 100 vs 400?
walks = np.cumsum(rng.normal(0, 1, (2000, 400)), axis=1)
show("2000 walks: spread at step 100 / 400", f"{walks[:, 99].std():.1f} / {walks[:, 399].std():.1f}")

# ---------------------------------------------------------------------------
# 2. A unit-root test from scratch (augmented Dickey-Fuller)
# ---------------------------------------------------------------------------
print("\n2. Augmented Dickey-Fuller, by hand")


def ols(X, yv):
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    s2 = e @ e / (len(yv) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se, e


def adf_t(x, lags=0):
    """t-statistic on gamma in  dx_t = c + gamma x_{t-1} + sum_k a_k dx_{t-k} + e.
    gamma = 0 means a unit root (a random walk); gamma < 0 pulls back to a mean."""
    dx = np.diff(x)
    rows = range(lags, len(dx))
    X = np.array([[1.0, x[t]] + [dx[t - k] for k in range(1, lags + 1)] for t in rows])
    b, se, _ = ols(X, dx[lags:])
    return b[1] / se[1]


# The catch: under the null, this t-stat is NOT normal. Simulate the null.
null_t = np.array([adf_t(np.cumsum(rng.normal(0, 1, 300))) for _ in range(2000)])
crit = np.quantile(null_t, 0.05)
show("5% critical value: simulated / normal table", f"{crit:.2f} / -1.64")
show("random walks 'rejected' using -1.64", f"{(null_t < -1.645).mean():.1%}  (should be 5%)")

# Two ways to make the rides stationary:
#  (a) weekly difference: this Monday minus last Monday;
#  (b) regress out what we know in advance (straight-line trend, weekday,
#      public holiday) and keep the residual.
md = np.array([str(d)[5:] for d in dates])
HOL = np.isin(md, HOLIDAYS)


def calendar(start, stop):
    """Known-in-advance columns: constant, trend, 6 weekday dummies, holiday."""
    t = np.arange(start, stop)
    return np.column_stack([np.ones(len(t)), t / 700] + [(t % 7 == k) for k in range(1, 7)]
                           + [HOL[start:stop]]).astype(float)


z7 = y[7:] - y[:-7]
H = 28  # keep lesson 011's last 28 days out of everything we fit
_, _, r = ols(calendar(0, len(y) - H), y[:-H])
for name, s, lags in [("AR(1), phi = 0.7", ar1, 0), ("random walk", walk, 0),
                      ("daily rides, raw", y, 7), ("rides, weekly difference", z7, 7),
                      ("rides minus trend/weekday/holiday", r, 7)]:
    t = adf_t(s, lags)
    show(f"ADF t: {name}", f"{t:6.2f}  {'stationary' if t < crit else 'unit root not ruled out'}")

# ---------------------------------------------------------------------------
# 3. An autoregression on what's left
# ---------------------------------------------------------------------------
print("\n3. AR(p) on the residual, order chosen by AIC")


def lagmat(z, p):
    """Rows t = p..n-1 with columns [1, z_{t-1}, ..., z_{t-p}]."""
    return np.column_stack([np.ones(len(z) - p)] + [z[p - k:len(z) - k] for k in range(1, p + 1)])


def aic(z, p, pmax):
    # Same rows for every p (drop the first pmax), so the AICs are comparable.
    zz = z[pmax - p:]
    e = ols(lagmat(zz, p), zz[p:])[2]
    return len(e) * np.log(e @ e / len(e)) + 2 * (p + 1)


def pick_p(z, pmax):
    return int(np.argmin([aic(z, p, pmax) for p in range(1, pmax + 1)])) + 1


P = pick_p(r, 14)
b, _, e = ols(lagmat(r, P), r[P:])
show("AIC picks p (residual, up to 14)", P)
show("AR coefficients, lag 1..p", " ".join(f"{v:+.2f}" for v in b[1:]))
show("sd: residual / one-step AR error", f"{r.std():.1f} / {e.std():.1f}")
P7 = pick_p(z7[:-H], 21)
show("AIC picks p (weekly difference, up to 21)", f"{P7}  (the maximum allowed)")


def run_ar(z, p, h):
    """Fit AR(p) to z, forecast h steps, feeding forecasts back in."""
    b = ols(lagmat(z, p), z[p:])[0]
    z = list(z)
    for _ in range(h):
        z.append(b[0] + sum(b[k] * z[-k] for k in range(1, p + 1)))
    return np.array(z[-h:])


def fc_reg_ar(hist, h, p):
    """Calendar regression, plus an AR(p) on its residual (p=0: regression only)."""
    o = len(hist)
    beta, _, res = ols(calendar(0, o), hist)
    base = calendar(o, o + h) @ beta
    return base + (run_ar(res, p, h) if p else 0)


def fc_diff_ar(hist, h, p):
    """AR(p) on weekly differences, then undo the differencing day by day."""
    zhat = run_ar(hist[7:] - hist[:-7], p, h)
    yy = list(hist)
    for zh in zhat:
        yy.append(yy[-7] + zh)
    return np.array(yy[-h:])


def fc_snaive(hist, h):
    return hist[-7:][np.arange(h) % 7]


train, test = y[:-H], y[-H:]
for name, f in [("seasonal naive", fc_snaive(train, H)),
                (f"weekly difference + AR({P7})", fc_diff_ar(train, H, P7)),
                (f"calendar regression + AR({P})", fc_reg_ar(train, H, P))]:
    show(f"011 holdout MAE: {name}", f"{np.abs(test - f).mean():.1f}")

# ---------------------------------------------------------------------------
# 4. A holdout that cheats
# ---------------------------------------------------------------------------
print("\n4. Random split vs the future, for lesson 011's degree-8 curve")
t_all = np.arange(len(y))


def poly8(t):
    tn = t / 700
    return np.column_stack([tn ** k for k in range(9)] + [(t % 7 == k) for k in range(1, 7)]).astype(float)


folds = rng.permutation(len(y)) % 5
err_rand = []
for f in range(5):
    tr, te = t_all[folds != f], t_all[folds == f]
    beta = np.linalg.lstsq(poly8(tr), y[tr], rcond=None)[0]
    err_rand.append(np.abs(y[te] - poly8(te) @ beta))
show("degree-8 curve: random 5-fold CV MAE", f"{np.concatenate(err_rand).mean():.1f}")

# ---------------------------------------------------------------------------
# 5. Rolling-origin evaluation
# ---------------------------------------------------------------------------
HZ, STEP, N_ORIG = 14, 14, 13
print(f"\n5. Rolling origin: {N_ORIG} origins, {HZ} days ahead each")
origins = [len(y) - HZ - STEP * i for i in range(N_ORIG)][::-1]
models = {
    "seasonal naive": lambda hist: fc_snaive(hist, HZ),
    "degree-8 curve": lambda hist: poly8(np.arange(len(hist), len(hist) + HZ))
    @ np.linalg.lstsq(poly8(np.arange(len(hist))), hist, rcond=None)[0],
    f"weekly difference + AR({P7})": lambda hist: fc_diff_ar(hist, HZ, P7),
    "calendar regression only": lambda hist: fc_reg_ar(hist, HZ, 0),
    f"calendar regression + AR({P})": lambda hist: fc_reg_ar(hist, HZ, P),
}
# Each origin: fit on everything before it, forecast the next 14 days, score.
err = {name: np.array([np.abs(y[o:o + HZ] - fc(y[:o])) for o in origins]) for name, fc in models.items()}
show("first origin / last origin", f"{dates[origins[0]]} / {dates[origins[-1]]}")
for name, E in err.items():
    show(name, f"MAE {E.mean():6.1f} | day 1 / 7 / 14: {E[:, 0].mean():5.1f} / {E[:, 6].mean():5.1f} / {E[:, 13].mean():5.1f}")
best = f"calendar regression + AR({P})"
wins = (err[best].mean(1) < err["seasonal naive"].mean(1)).sum()
show("origins where it beats seasonal naive", f"{wins} of {N_ORIG}")

# ---------------------------------------------------------------------------
# Figure: the residual we modelled, and error by horizon under the rolling origin
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(dates[:len(r)], r, lw=0.6, color="C2")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_title("Rides minus trend, weekday and holidays")
ax[0].tick_params(axis="x", labelsize=7)
for name, E in err.items():
    if name != "degree-8 curve":
        ax[1].plot(np.arange(1, HZ + 1), E.mean(0), "o-", ms=3, label=name)
ax[1].set_xlabel("days ahead")
ax[1].set_ylabel("MAE (rides), mean over origins")
ax[1].set_title("Rolling origin: error by horizon")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "rolling_origin.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/rolling_origin.png")
