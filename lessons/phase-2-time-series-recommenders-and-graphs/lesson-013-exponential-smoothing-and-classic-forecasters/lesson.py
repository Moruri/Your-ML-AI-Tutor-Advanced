"""
Lesson 013 - Exponential smoothing and classic forecasters

Lesson 012 built an autoregression and a rolling-origin test. Today we
meet the forecasters that ran most of the world's demand planning for
fifty years: simple exponential smoothing, Holt's trend, and Holt-Winters
with a weekly season. We write each one as a few update rules, fit its
smoothing weights on the past only, and race them against the seasonal
naive with the same rolling origin as lesson 012.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/holt_winters.png`,
next to this script (ignored by git), and runs in under a minute.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def show(label, value):
    print(f"  {label:<46} -> {value}")


# Same series as lessons 011 and 012: same function, same seed.
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
rng = np.random.default_rng(13)

# ---------------------------------------------------------------------------
# 1. Simple exponential smoothing: an average that forgets
# ---------------------------------------------------------------------------
print("\n1. Simple exponential smoothing (SES)")


def ses(x, alpha):
    """level_t = alpha * x_t + (1 - alpha) * level_{t-1}. Returns one-step forecasts
    (the level BEFORE seeing each point) and the final level."""
    level, fc = x[0], np.empty(len(x))
    for i, v in enumerate(x):
        fc[i] = level
        level = alpha * v + (1 - alpha) * level
    return fc, level


for a in (0.1, 0.5):
    w = a * (1 - a) ** np.arange(5)
    half = np.log(0.5) / np.log(1 - a)
    show(f"alpha = {a}: weights on last 5 points", " ".join(f"{v:.3f}" for v in w) + f"  (half-life {half:.1f})")

# A toy level that jumps from 100 to 130 at step 150. Noise sd 10.
toy = np.r_[np.full(150, 100.0), np.full(150, 130.0)] + rng.normal(0, 10, 300)
for a in (0.1, 0.5):
    fc, _ = ses(toy, a)
    steady = np.abs(toy[50:150] - fc[50:150]).mean()
    catch = np.argmax(np.abs(fc[150:] - 130) < 5)
    show(f"alpha = {a}: steady MAE / days to catch up", f"{steady:.1f} / {catch}")
sse = lambda a: ((toy - ses(toy, a)[0])[1:] ** 2).sum()
a_hat = minimize(lambda p: sse(p[0]), [0.3], bounds=[(0.01, 1)]).x[0]
show("alpha fitted by one-step squared error", f"{a_hat:.2f}")

# ---------------------------------------------------------------------------
# 2 and 3. Holt-Winters: level + (damped) trend + weekly season
# ---------------------------------------------------------------------------
M = 7


def hw_filter(x, a, b, g, phi=1.0, trend=True, season=True):
    """Additive Holt-Winters. Returns one-step errors and the final state.
    Start: level = mean of week 1, trend = (week 2 - week 1) / 7, season = week 1 - level."""
    level = x[:M].mean()
    slope = (x[M:2 * M].mean() - level) / M if trend else 0.0
    s = list(x[:M] - level) if season else [0.0] * M
    err = np.empty(len(x))
    for i, v in enumerate(x):
        si = s[i % M]
        pred = level + phi * slope + si
        err[i] = v - pred
        new_level = a * (v - si) + (1 - a) * (level + phi * slope)
        if trend:
            slope = b * (new_level - level) + (1 - b) * phi * slope
        if season:
            s[i % M] = g * (v - new_level) + (1 - g) * si
        level = new_level
    return err, (level, slope, s, len(x))


def hw_forecast(state, h, phi=1.0):
    level, slope, s, n = state
    damp = np.cumsum(phi ** np.arange(1, h + 1))
    return np.array([level + damp[k] * slope + s[(n + k) % M] for k in range(h)])


SPECS = {  # name: (trend, season, damped)
    "SES": (False, False, False),
    "Holt-Winters": (True, True, False),
    "Holt-Winters, damped": (True, True, True),
}


def fit(x, trend, season, damped, start=None):
    """Choose smoothing weights by minimising one-step squared error on x only."""
    def unpack(p):
        a, b, g = p[0], p[1] if trend else 0, p[2] if season else 0
        return a, b, g, (p[3] if damped else 1.0)

    def loss(p):
        a, b, g, phi = unpack(p)
        e = hw_filter(x, a, b, g, phi, trend, season)[0][2 * M:]
        return e @ e / len(e)

    p0 = start if start is not None else [0.3, 0.05, 0.1, 0.9]
    bounds = [(0.01, 1), (0.0, 0.5), (0.0, 0.5), (0.8, 0.995)]
    p = minimize(loss, p0, bounds=bounds, method="L-BFGS-B").x
    a, b, g, phi = unpack(p)
    err, state = hw_filter(x, a, b, g, phi, trend, season)
    return dict(a=a, b=b, g=g, phi=phi, err=err[2 * M:], state=state, p=p)


H = 28  # lesson 011's holdout: keep the last 28 days out of the fit
train, test = y[:-H], y[-H:]
print("\n2. Holt and the damped trend, fitted on the first 672 days")
fits = {name: fit(train, *spec) for name, spec in SPECS.items()}
for name in ["Holt-Winters", "Holt-Winters, damped"]:
    f = fits[name]
    show(f"{name.replace('Holt-Winters', 'HW')}: alpha, beta, gamma, phi", f"{f['a']:.2f}, {f['b']:.3f}, {f['g']:.2f}, {f['phi']:.3f}")
for name in ["Holt-Winters", "Holt-Winters, damped"]:
    f = fits[name]
    fc = hw_forecast(f["state"], 365, f["phi"])
    show(f"{name.replace('Holt-Winters', 'HW')}: slope / gain over 1 year", f"{f['state'][1]:+.2f} / {fc[-7:].mean() - fc[:7].mean():+.0f} rides")
show("truth: slope / gain over 1 year", f"+0.55 / {0.55 * 358:+.0f} rides")

print("\n3. Holt-Winters on the rides")
hw = fits["Holt-Winters"]
s_now = np.array(hw["state"][2])
# Season slot i % 7 belongs to weekday i % 7, and day 0 (2024-01-01) is a Monday.
show("learned weekly pattern, Mon..Sun", " ".join(f"{v:+.0f}" for v in s_now - s_now.mean()))
show("true weekly pattern, Mon..Sun", "+60 +75 +80 +75 +70 -110 -250")
show("one-step error sd: SES / HW / HW damped",
     " / ".join(f"{fits[n]['err'].std():.1f}" for n in SPECS))
for name in SPECS:
    f = fits[name]
    show(f"011 holdout MAE: {name}", f"{np.abs(test - hw_forecast(f['state'], H, f['phi'])).mean():.1f}")
show("011 holdout MAE: seasonal naive (lesson 011)", f"{np.abs(test - train[-7:][np.arange(H) % 7]).mean():.1f}")

# ---------------------------------------------------------------------------
# 4. A fair race: lesson 012's rolling origin
# ---------------------------------------------------------------------------
HOL = np.isin(np.array([str(d)[5:] for d in dates]), HOLIDAYS)


def ols(X, yv):
    return np.linalg.lstsq(X, yv, rcond=None)[0]


def calendar(start, stop):
    t = np.arange(start, stop)
    return np.column_stack([np.ones(len(t)), t / 700] + [(t % 7 == k) for k in range(1, 7)]
                           + [HOL[start:stop]]).astype(float)


def fc_reg_ar(hist, h, p=6):
    """Lesson 012's winner: calendar regression + AR(p) on its residual."""
    o = len(hist)
    beta = ols(calendar(0, o), hist)
    r = hist - calendar(0, o) @ beta
    X = np.column_stack([np.ones(len(r) - p)] + [r[p - k:len(r) - k] for k in range(1, p + 1)])
    c = ols(X, r[p:])
    z = list(r)
    for _ in range(h):
        z.append(c[0] + sum(c[k] * z[-k] for k in range(1, p + 1)))
    return calendar(o, o + h) @ beta + np.array(z[-h:])


HZ, STEP, N_ORIG = 14, 14, 13
origins = [len(y) - HZ - STEP * i for i in range(N_ORIG)][::-1]
print(f"\n4. Rolling origin: {N_ORIG} origins, {HZ} days ahead, every model refitted each time")
err = {"seasonal naive": [], **{n: [] for n in SPECS}, "calendar regression + AR(6)": [],
       "average of HW damped and reg + AR": []}
paths_cover = {"flat +/- 1.28 sd": [], "simulated paths": []}
starts = {n: None for n in SPECS}
for o in origins:
    hist, truth = y[:o], y[o:o + HZ]
    err["seasonal naive"].append(np.abs(truth - hist[-7:][np.arange(HZ) % 7]))
    for name, spec in SPECS.items():
        f = fit(hist, *spec, start=starts[name])
        starts[name] = f["p"]  # warm start the next origin
        fc = hw_forecast(f["state"], HZ, f["phi"])
        err[name].append(np.abs(truth - fc))
        if name == "Holt-Winters, damped":
            hwd_fc, hwd = fc, f
    ra = fc_reg_ar(hist, HZ)
    err["calendar regression + AR(6)"].append(np.abs(truth - ra))
    err["average of HW damped and reg + AR"].append(np.abs(truth - (hwd_fc + ra) / 2))

    # 5. Intervals for HW damped: flat width vs simulated future paths.
    sd = hwd["err"].std()
    paths_cover["flat +/- 1.28 sd"].append(np.abs(truth - hwd_fc) < 1.28 * sd)
    sims = []
    for _ in range(500):
        e = rng.choice(hwd["err"], HZ)  # bootstrap past one-step errors
        level, slope, s, n = hwd["state"]
        s, path = list(s), []
        for k in range(HZ):
            si = s[(n + k) % M]
            v = level + hwd["phi"] * slope + si + e[k]
            new_level = hwd["a"] * (v - si) + (1 - hwd["a"]) * (level + hwd["phi"] * slope)
            slope = hwd["b"] * (new_level - level) + (1 - hwd["b"]) * hwd["phi"] * slope
            s[(n + k) % M] = hwd["g"] * (v - new_level) + (1 - hwd["g"]) * si
            level = new_level
            path.append(v)
        sims.append(path)
    lo, hi = np.quantile(np.array(sims), [0.1, 0.9], axis=0)
    paths_cover["simulated paths"].append((truth >= lo) & (truth <= hi))
    if o == origins[-1]:
        last = dict(o=o, fc=hwd_fc, lo=lo, hi=hi, sd=sd)

err = {k: np.array(v) for k, v in err.items()}
for name, E in err.items():
    show(name, f"MAE {E.mean():5.1f} | day 1 / 7 / 14: {E[:, 0].mean():5.1f} / {E[:, 6].mean():5.1f} / {E[:, 13].mean():5.1f}")
for name in ["Holt-Winters, damped", "calendar regression + AR(6)"]:
    wins = (err[name].mean(1) < err["seasonal naive"].mean(1)).sum()
    short = name.replace("Holt-Winters", "HW").replace("calendar regression", "calendar reg.")
    show(f"beats seasonal naive: {short}", f"{wins} of {N_ORIG} origins")

# ---------------------------------------------------------------------------
# 5. How wide is the fog? 80% intervals, checked
# ---------------------------------------------------------------------------
print("\n5. 80% intervals for HW damped: share of real days inside")
for name, C in paths_cover.items():
    C = np.array(C)
    show(name, f"all {C.mean():.0%} | days 1-3 {C[:, :3].mean():.0%} | days 12-14 {C[:, -3:].mean():.0%}")
show("last origin: width day 1 / day 14 (simulated)", f"{last['hi'][0] - last['lo'][0]:.0f} / {last['hi'][-1] - last['lo'][-1]:.0f} rides")

# ---------------------------------------------------------------------------
# Figure: the last origin, and error by horizon
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
o = last["o"]
ax[0].plot(dates[o - 42:o + HZ], y[o - 42:o + HZ], "k.-", lw=0.6, ms=3, label="actual")
ax[0].plot(dates[o:o + HZ], last["fc"], "C1-", label="HW damped")
ax[0].fill_between(dates[o:o + HZ], last["lo"], last["hi"], color="C1", alpha=0.25, label="80% (simulated)")
ax[0].axvline(dates[o], color="grey", lw=0.6)
ax[0].set_title("Last origin: forecast and interval")
ax[0].tick_params(axis="x", labelsize=7, rotation=30)
ax[0].legend(fontsize=8)
for name, E in err.items():
    if name == "SES":
        continue  # off the scale; see the printed table
    ax[1].plot(np.arange(1, HZ + 1), E.mean(0), "o-", ms=3, label=name)
ax[1].set_xlabel("days ahead")
ax[1].set_ylabel("MAE (rides), mean over origins")
ax[1].set_title("Rolling origin: error by horizon")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "holt_winters.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/holt_winters.png")
