"""
Lesson 011 - Time series as a modelling problem

Phase 1 was about rows that didn't care about each other's order. Now the
order is the point. We take two years of daily rides across the network,
pull them apart into trend, weekly pattern and what's left, and then set
the bar every later forecaster in this phase has to clear: a few dumb
baselines that turn out to be surprisingly hard to beat.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/decomposition.png`,
next to this script (ignored by git), and runs in about a second.
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


# ---------------------------------------------------------------------------
# The data: 700 days of network-wide rides, 2024-01-01 (a Monday) onwards.
# Simulated so we know the truth. The same function, with the same seed,
# appears in lesson 012, so both lessons see exactly the same series.
# ---------------------------------------------------------------------------
HOLIDAYS = ["01-01", "05-01", "06-01", "10-20", "12-12", "12-25", "12-26"]


def rides_series(n_days=700, seed=11):
    """Daily rides: growing trend + weekly pattern + rainy seasons + holidays
    + sticky noise. Returns (dates, rides)."""
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2024-01-01") + np.arange(n_days)
    t = np.arange(n_days)
    dow = t % 7  # 0 = Monday
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    trend = 900 + 0.55 * t  # the network keeps growing
    weekly = np.array([60, 75, 80, 75, 70, -110, -250])[dow]  # commuters; quiet Sundays
    # Nairobi's two rainy seasons: long rains (Apr-May), short rains (Nov)
    rain = -120 * np.exp(-0.5 * ((doy - 120) / 22) ** 2) - 70 * np.exp(-0.5 * ((doy - 320) / 15) ** 2)
    md = np.array([str(d)[5:] for d in dates])
    holiday = np.where(np.isin(md, HOLIDAYS), -300, 0)
    noise = np.zeros(n_days)
    for i in range(n_days):  # AR(1): a busy day tends to follow a busy day
        noise[i] = (0.6 * noise[i - 1] if i else 0) + rng.normal(0, 35)
    return dates, np.round(trend + weekly + rain + holiday + noise)


dates, y = rides_series()
DOW = np.arange(len(y)) % 7
NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---------------------------------------------------------------------------
# 1. Order is information
# ---------------------------------------------------------------------------
print("\n1. Order is information")
show("days / first / last", f"{len(y)} / {dates[0]} / {dates[-1]}")
show("mean rides, first 8 weeks / last 8 weeks", f"{y[:56].mean():.0f} / {y[-56:].mean():.0f}")


def autocorr(x, lag):
    x = x - x.mean()
    return (x[:-lag] * x[lag:]).sum() / (x * x).sum()


shuffled = np.random.default_rng(0).permutation(y)
show("correlation with yesterday: real / shuffled", f"{autocorr(y, 1):.2f} / {autocorr(shuffled, 1):+.2f}")
show("correlation with a week ago: real / shuffled", f"{autocorr(y, 7):.2f} / {autocorr(shuffled, 7):+.2f}")

# ---------------------------------------------------------------------------
# 2. Decompose: trend-cycle + weekly pattern + remainder
# ---------------------------------------------------------------------------
print("\n2. Classical decomposition (additive, period 7)")


def decompose(x, period=7):
    """Centred moving average for the trend-cycle, weekday means of the
    detrended series for the seasonal part, and whatever is left over."""
    k = period // 2
    trend = np.full(len(x), np.nan)
    trend[k:-k] = np.convolve(x, np.ones(period) / period, mode="valid")
    detr = x - trend
    idx = np.array([np.nanmean(detr[d::period]) for d in range(period)])
    idx -= idx.mean()  # the weekly pattern should average to zero
    season = idx[np.arange(len(x)) % period]
    return trend, season, x - trend - season, idx


trend, season, resid, idx = decompose(y)
show("weekly index, Mon..Sun", " ".join(f"{v:+.0f}" for v in idx))
ok = ~np.isnan(trend)
v = y[ok].var()
show("share of variance: trend / weekly / remainder",
     f"{trend[ok].var() / v:.0%} / {season[ok].var() / v:.0%} / {resid[ok].var() / v:.0%}")
show("trend-cycle values missing at each end", f"{(~ok[:10]).sum()} days")
show("remainder: correlation with yesterday", f"{autocorr(resid[ok], 1):.2f}")
worst = np.argsort(resid[ok])[:3]
show("three most negative remainders fall on", ", ".join(str(d) for d in np.sort(dates[ok][worst])))

# ---------------------------------------------------------------------------
# 3. Baselines: the forecasts everything else must beat
# ---------------------------------------------------------------------------
print("\n3. Baselines, forecasting the last 28 days")
H = 28
train, test = y[:-H], y[-H:]
n = len(train)
h = np.arange(1, H + 1)

forecasts = {
    "mean": np.full(H, train.mean()),
    "naive (last value)": np.full(H, train[-1]),
    "drift": train[-1] + h * (train[-1] - train[0]) / (n - 1),
    "seasonal naive (last week)": train[-7:][(h - 1) % 7],
    "seasonal mean (last 4 weeks)": train[-28:].reshape(4, 7).mean(0)[(h - 1) % 7],
}
# Note what the decomposition can't do at the forecast origin: its centred
# trend needs 3 days of the future, so the last 3 training days have none.
tr_trend, _, _, tr_idx = decompose(train)
last = np.where(~np.isnan(tr_trend))[0][-1]
forecasts["decomposition (flat trend + index)"] = tr_trend[last] + tr_idx[(n + h - 1) % 7]

# In-sample MAE of the seasonal naive: the yardstick for MASE.
scale = np.abs(train[7:] - train[:-7]).mean()


def score(f):
    e = test - f
    return np.abs(e).mean(), np.sqrt((e ** 2).mean()), np.abs(e).mean() / scale


for name, f in forecasts.items():
    mae, rmse, mase = score(f)
    show(name, f"MAE {mae:5.1f}  RMSE {rmse:5.1f}  MASE {mase:.2f}")
show("MASE scale: in-sample seasonal-naive MAE", f"{scale:.1f}")

# ---------------------------------------------------------------------------
# 4. A clever model that loses
# ---------------------------------------------------------------------------
print("\n4. A flexible curve, fitted to all the history")
tt = np.arange(n) / n


def design(tn, d, deg):
    return np.column_stack([tn ** p for p in range(deg + 1)] + [(d == k) for k in range(1, 7)]).astype(float)


for deg in (1, 8):
    beta = np.linalg.lstsq(design(tt, DOW[:n], deg), train, rcond=None)[0]
    fit_in = design(tt, DOW[:n], deg) @ beta
    f = design((n + h - 1) / n, DOW[n:], deg) @ beta
    in_mae = np.abs(train - fit_in).mean()
    show(f"poly degree {deg} + weekday: in-sample / test MAE", f"{in_mae:.1f} / {score(f)[0]:.1f}")
    forecasts[f"poly degree {deg} + weekday"] = f

# ---------------------------------------------------------------------------
# Figure: series, trend, weekly pattern, remainder, and the forecasts
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].plot(dates, y, lw=0.5, color="grey", label="daily rides")
ax[0, 0].plot(dates, trend, color="C0", label="trend-cycle (7-day centred mean)")
ax[0, 0].set_title("The series and its trend-cycle")
ax[0, 0].legend(fontsize=8)
ax[0, 1].bar(NAMES, idx, color="C1")
ax[0, 1].set_title("Weekly index (rides vs the trend)")
ax[1, 0].plot(dates, resid, lw=0.6, color="C2")
ax[1, 0].axhline(0, color="k", lw=0.6)
ax[1, 0].set_title("Remainder: holidays and day-to-day noise")
win = slice(len(y) - 70, None)
ax[1, 1].plot(dates[win], y[win], "k.-", lw=0.8, label="actual")
for name in ["seasonal naive (last week)", "drift", "poly degree 8 + weekday"]:
    ax[1, 1].plot(dates[-H:], forecasts[name], lw=1.2, label=name)
ax[1, 1].axvline(dates[-H], color="grey", lw=0.8)
ax[1, 1].set_title("Forecasting the last 28 days")
ax[1, 1].legend(fontsize=8)
for a in ax.ravel():
    a.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "decomposition.png", dpi=120)
print(f"\nSaved {OUT.relative_to(HERE)}/decomposition.png")
