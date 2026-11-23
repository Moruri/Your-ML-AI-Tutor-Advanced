"""
Lesson 014 - Features, exogenous signals and when ML helps a forecast

Lessons 011-013 forecast the rides from the rides alone. Today we turn the
forecast into an ordinary tabular problem: one row per (day, lead time),
with calendar, lag and weather columns, and a gradient-boosted model on top.
The real lesson is the bookkeeping: which columns you would actually have
when the forecast is made, and when the clever model earns its place.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/features.png`, next to
this script (ignored by git), and runs in about half a minute.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny data: extra threads only add overhead
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def show(label, value):
    print(f"  {label:<46} -> {value}")


HOLIDAYS = ["01-01", "05-01", "06-01", "10-20", "12-12", "12-25", "12-26"]
N = 700


def wet_chance(doy):
    """Climatology: chance of a wet day, peaking in the long and short rains."""
    return 0.12 + 0.5 * np.exp(-0.5 * ((doy - 115) / 25) ** 2) + 0.4 * np.exp(-0.5 * ((doy - 320) / 18) ** 2)


def city(n_days=N, seed=14):
    """Same city as lessons 011-013, but the rain is now real daily rainfall
    instead of a smooth seasonal dip, and the stadium has match days.
    Returns everything the ops team could know, plus the truth."""
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2024-01-01") + np.arange(n_days)
    t = np.arange(n_days)
    dow = t % 7
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    md = np.array([str(d)[5:] for d in dates])
    hol = np.isin(md, HOLIDAYS)
    wet = rng.random(n_days) < wet_chance(doy)
    rain = np.where(wet, rng.gamma(0.8, 10, n_days), 0.0).round(1)  # mm
    # Matches: about one weekend day in four, plus the odd midweek game.
    match = ((dow >= 5) & (rng.random(n_days) < 0.25)) | ((dow == 2) & (rng.random(n_days) < 0.08))
    weekend = dow >= 5
    rain_hit = -200 * (1 - np.exp(-rain / 8)) * np.where(weekend, 1.7, 1.0)  # saturates; leisure riders mind more
    match_lift = np.where(match, np.where(rain < 2, 180, 40), 0)  # nobody cycles to a wet match
    weekly = np.array([60, 75, 80, 75, 70, -110, -250])[dow]
    noise = np.zeros(n_days)
    for i in range(n_days):
        noise[i] = (0.6 * noise[i - 1] if i else 0) + rng.normal(0, 35)
    y = np.round(900 + 0.55 * t + weekly + np.where(hol, -300, 0) + rain_hit + match_lift + noise)

    # The weather service's forecasts: rain_fc[d, h] is the forecast for day d
    # issued h days earlier (h = 1..7). Skill fades with lead time: with
    # probability q_h it's close to the truth, otherwise it's a climatology guess.
    rain_fc = np.zeros((n_days, 8))
    for h in range(1, 8):
        good = rng.random(n_days) < 0.92 - 0.05 * (h - 1)
        guess = np.where(rng.random(n_days) < wet_chance(doy), rng.gamma(0.8, 10, n_days), 0.0)
        rain_fc[:, h] = np.where(good, rain * rng.lognormal(0, 0.25, n_days), guess).round(1)
    return dict(dates=dates, y=y, dow=dow, doy=doy, hol=hol, match=match, rain=rain, rain_fc=rain_fc)


C = city()
dates, y = C["dates"], C["y"]

# ---------------------------------------------------------------------------
# 1. The new data, and what it's worth
# ---------------------------------------------------------------------------
print("\n1. Rides, rain and matches")
wet = C["rain"] >= 2
show("wet days (>= 2 mm) / match days", f"{wet.mean():.0%} / {C['match'].sum()}")
for name, mask in [("weekday", C["dow"] < 5), ("weekend", C["dow"] >= 5)]:
    dry, wt = y[mask & ~wet & ~C["match"] & ~C["hol"]], y[mask & wet & ~C["match"] & ~C["hol"]]
    show(f"{name}: mean rides dry / wet", f"{dry.mean():.0f} / {wt.mean():.0f}  (gap {dry.mean() - wt.mean():.0f})")
for h in (1, 3, 7):
    f = C["rain_fc"][:, h] >= 2
    show(f"rain forecast, {h} day(s) ahead: wet days caught", f"{(f & wet).sum() / wet.sum():.0%}, false alarms {(f & ~wet).sum() / f.sum():.0%}")

# ---------------------------------------------------------------------------
# 2. Turning a forecast into rows
# ---------------------------------------------------------------------------
# One row per (target day d, lead time h): "standing at day d - h, forecast day d".
# Every column must be known at d - h. Lags of 7 days or more are known for any
# h <= 7, the calendar is known forever, the rain is only known as a forecast.
H = 7
NAMES = ["trend", "dow", "holiday", "match", "doy_sin", "doy_cos", "lead", "lag7", "lag7_13_mean", "rain_fc"]


def rows(days, lead, rain_col="forecast", base=None, last=None):
    """Feature matrix for target days `days` at lead time `lead`.
    `base` is a trend to subtract from y-based columns (None: raw rides)."""
    b = base if base is not None else np.zeros(N)
    yy = y - b
    d = np.asarray(days)
    rain = C["rain_fc"][d, lead] if rain_col == "forecast" else C["rain"][d]
    cols = [d / N, C["dow"][d], C["hol"][d], C["match"][d],
            np.sin(2 * np.pi * C["doy"][d] / 365.25), np.cos(2 * np.pi * C["doy"][d] / 365.25),
            np.full(len(d), lead), yy[d - 7], np.mean([yy[d - k] for k in range(7, 14)], axis=0), rain]
    if last == "honest":
        cols.append(yy[d - lead])  # the newest day you'd actually have
    elif last == "leak":
        cols.append(yy[d - 1])  # only exists when lead == 1
    return np.column_stack(cols).astype(float)


def stack(days, **kw):
    X = np.vstack([rows(days, h, **kw) for h in range(1, H + 1)])
    return X, np.tile(np.asarray(days), H)


print("\n2. Rows for a direct, multi-lead model")
o0 = N - H - 7 * 25  # the first origin used in section 3
X0, d0 = stack(np.arange(14, o0))
show(f"training rows at the first origin, {dates[o0]}", f"{X0.shape[0]} rows = {o0 - 14} days x 7 leads")
d = int(np.where(dates == np.datetime64("2025-11-29"))[0][0])
ex = rows([d], 3)[0]
show("one row: Sat 2025-11-29 at lead 3", "made on Wed 2025-11-26, knowing:")
for n, v in zip(NAMES, ex):
    print(f"      {n:<14}{v:9.2f}")

# ---------------------------------------------------------------------------
# Models. Each takes the history up to origin o and returns 7 forecasts.
# ---------------------------------------------------------------------------


def onehot(X, names):
    """For least squares: weekday dummies instead of a 0-6 code, and no lead column."""
    dow = X[:, names.index("dow")].astype(int)
    other = [k for k, n in enumerate(names) if n not in ("dow", "lead")]
    return np.column_stack([np.ones(len(X)), X[:, other]] + [(dow == k) for k in range(1, 7)]).astype(float)


def gbm():
    # Small, shallow trees: a few hundred days is not much data. If you tune
    # these, tune inside each origin, never on the test windows.
    return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=40, random_state=0)


def line_trend(o):
    """Straight line fitted to the days before the origin, extended over all days."""
    b = np.polyfit(np.arange(o), y[:o], 1)
    return np.polyval(b, np.arange(N))


def fc_snaive(o):
    return y[o - 7:o]


def fc_reg_ar(o, p=6):
    """Lesson 012's calendar regression + AR(6): knows holidays, not rain or matches."""
    def cal(a, z):
        t = np.arange(a, z)
        return np.column_stack([np.ones(len(t)), t / N] + [(t % 7 == k) for k in range(1, 7)] + [C["hol"][a:z]]).astype(float)
    beta = np.linalg.lstsq(cal(0, o), y[:o], rcond=None)[0]
    r = y[:o] - cal(0, o) @ beta
    X = np.column_stack([np.ones(o - p)] + [r[p - k:o - k] for k in range(1, p + 1)])
    c = np.linalg.lstsq(X, r[p:], rcond=None)[0]
    z = list(r)
    for _ in range(H):
        z.append(c[0] + sum(c[k] * z[-k] for k in range(1, p + 1)))
    return cal(o, o + H) @ beta + np.array(z[-H:])


def fc_table(o, kind, rain_col="forecast", detrend=True, drop=(), last=None):
    """kind: "linear" (least squares) or "trees" (gradient boosting).
    Training rows: every target day before the origin, at every lead.
    (Target day d at lead h only uses rides up to d - h, so nothing after o leaks in.)"""
    base = line_trend(o) if detrend else np.zeros(N)
    kw = dict(rain_col=rain_col, base=base, last=last)
    Xtr, dtr = stack(np.arange(14, o), **kw)
    target = y[dtr] - base[dtr]
    test_days = np.arange(o, o + H)
    Xte = np.vstack([rows(test_days[h - 1:h], h, **kw) for h in range(1, H + 1)])
    names = NAMES + (["last"] if last else [])
    keep = [k for k, n in enumerate(names) if n not in drop]
    names = [names[k] for k in keep]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if kind == "linear":
        m = np.linalg.lstsq(onehot(Xtr, names), target, rcond=None)[0]
        pred = onehot(Xte, names) @ m
    else:
        pred = gbm().set_params(categorical_features=[names.index("dow")]).fit(Xtr, target).predict(Xte)
    return pred + base[test_days]


MODELS = {
    "seasonal naive": fc_snaive,
    "calendar regression + AR(6)": fc_reg_ar,
    "linear, all features": lambda o: fc_table(o, "linear"),
    "boosted trees, raw rides": lambda o: fc_table(o, "trees", detrend=False),
    "boosted trees, detrended": lambda o: fc_table(o, "trees"),
    "boosted trees, no rain column": lambda o: fc_table(o, "trees", drop=("rain_fc",)),
    "boosted trees, actual rain (oracle)": lambda o: fc_table(o, "trees", rain_col="actual"),
}


def race(names, n_orig=26):
    """Weekly origins over the last n_orig weeks, 7 days ahead, refitting every time."""
    origins = [N - H - 7 * i for i in range(n_orig)][::-1]
    T = np.array([y[o:o + H] for o in origins])
    out = {}
    for name in names:
        P = np.array([MODELS[name](o) if isinstance(name, str) else name[1](o) for o in origins])
        out[name if isinstance(name, str) else name[0]] = (np.abs(T - P), (P - T).mean())
    return origins, out


def report(out, bias=True):
    for name, (E, b) in out.items():
        extra = f" | bias {b:+5.1f}" if bias else ""
        show(name, f"MAE {E.mean():5.1f} | day 1 / 7: {E[:, 0].mean():5.1f} / {E[:, 6].mean():5.1f}{extra}")


# ---------------------------------------------------------------------------
# 3. The race: weekly origins over the last 26 weeks, 7 days ahead each
# ---------------------------------------------------------------------------
print("\n3. Rolling origin: 26 weekly origins, 7 days ahead, every model refitted")
origins, res = race(list(MODELS))
show("first origin / last origin", f"{dates[origins[0]]} / {dates[origins[-1]]}")
report(res)
test_idx = np.concatenate([np.arange(o, o + H) for o in origins])
wet_t = C["rain"][test_idx] >= 2
for name in ["calendar regression + AR(6)", "linear, all features", "boosted trees, detrended"]:
    E = res[name][0].ravel()
    show(f"MAE dry / wet days: {name}", f"{E[~wet_t].mean():.1f} / {E[wet_t].mean():.1f}")

# ---------------------------------------------------------------------------
# 4. Two leaks that flatter a forecast
# ---------------------------------------------------------------------------
print("\n4. Leaks: yesterday's rides, and tomorrow's weather")
_, leak = race([("trees + newest known day (lag = lead)", lambda o: fc_table(o, "trees", last="honest")),
                ("trees + yesterday at every lead (LEAK)", lambda o: fc_table(o, "trees", last="leak"))])
report(leak, bias=False)
show("trees with actual rain vs rain forecast",
     f"{res['boosted trees, actual rain (oracle)'][0].mean():.1f} vs {res['boosted trees, detrended'][0].mean():.1f}")

# ---------------------------------------------------------------------------
# 5. When do the trees help?
# ---------------------------------------------------------------------------


def rides_series(n_days=700, seed=11):
    """Lessons 011-013's rides: smooth rainy-season dips, no daily rain, no matches."""
    rng = np.random.default_rng(seed)
    d = np.datetime64("2024-01-01") + np.arange(n_days)
    t = np.arange(n_days)
    doy = (d - d.astype("datetime64[Y]")).astype(int)
    md = np.array([str(v)[5:] for v in d])
    rain = -120 * np.exp(-0.5 * ((doy - 120) / 22) ** 2) - 70 * np.exp(-0.5 * ((doy - 320) / 15) ** 2)
    noise = np.zeros(n_days)
    for i in range(n_days):
        noise[i] = (0.6 * noise[i - 1] if i else 0) + rng.normal(0, 35)
    return np.round(900 + 0.55 * t + np.array([60, 75, 80, 75, 70, -110, -250])[t % 7]
                    + rain + np.where(np.isin(md, HOLIDAYS), -300, 0) + noise)


THREE = ["calendar regression + AR(6)", "linear, all features", "boosted trees, detrended"]
saved = (y, C, N)  # the helpers read these globals; swap in other worlds, then restore

print("\n5a. Lesson 011-013's rides: nothing nonlinear to find")
y, C = rides_series(), dict(C, match=np.zeros(N, bool), rain=np.zeros(N), rain_fc=np.zeros((N, 8)))
report(race(THREE)[1], bias=False)

for n_days in (1400, 2800):
    print(f"\n5b. The same rainy city, with {n_days} days of history")
    N = n_days
    C = city(n_days)
    y = C["y"]
    report(race(THREE)[1], bias=False)
y, C, N = saved

# ---------------------------------------------------------------------------
# Figure: error on dry and wet days, and the last two weeks of forecasts
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
show_models = ["seasonal naive", "calendar regression + AR(6)", "linear, all features", "boosted trees, detrended"]
xs = np.arange(len(show_models))
ax[0].bar(xs - 0.2, [res[m][0].ravel()[~wet_t].mean() for m in show_models], 0.4, label="dry days")
ax[0].bar(xs + 0.2, [res[m][0].ravel()[wet_t].mean() for m in show_models], 0.4, label="wet days")
ax[0].set_xticks(xs, ["seasonal\nnaive", "calendar\nreg + AR", "linear,\nfeatures", "boosted\ntrees"], fontsize=8)
ax[0].set_ylabel("MAE (rides)")
ax[0].set_title("Rolling origin: where the error comes from")
ax[0].legend(fontsize=8)
o1, o2 = origins[-2], origins[-1]
ax[1].plot(dates[o1 - 14:o2 + H], y[o1 - 14:o2 + H], "k.-", lw=0.6, ms=3, label="actual")
for o in (o1, o2):
    for m, col in [("calendar regression + AR(6)", "C1"), ("boosted trees, detrended", "C2")]:
        ax[1].plot(dates[o:o + H], MODELS[m](o), color=col, label=m if o == o2 else None)
for d in np.where(C["rain"][o1 - 14:o2 + H] >= 2)[0]:
    ax[1].axvspan(dates[o1 - 14 + d] - np.timedelta64(12, "h"), dates[o1 - 14 + d] + np.timedelta64(12, "h"), color="C0", alpha=0.12)
ax[1].set_title("Last two origins (blue bands: wet days)")
ax[1].tick_params(axis="x", labelsize=7, rotation=30)
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "features.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/features.png")
