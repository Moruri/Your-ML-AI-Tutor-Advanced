"""
Lesson 021 - Attention from first principles

A dispatcher at the depot gets a stream of radio reports: "station so-and-so,
N bikes". Then someone asks: "how many bikes at the station by the market?"
To answer, you have to find the right report in the stream and read its number.
That is a lookup, and attention is a lookup you can differentiate.

We build it from nothing in NumPy: a soft dictionary lookup, the softmax and
why we scale by sqrt(d), self-attention over a sequence with a causal mask,
then a lookup task where the query and the keys speak different "languages"
and the model has to learn the translation, with gradients written by hand
and checked. We finish with what attention cannot see (order) and what it
costs (n squared).

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/attention.png`, next to
this script (ignored by git), and runs in a few seconds on a laptop.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
np.set_printoptions(precision=3, suppress=True)


def show(label, value):
    print(f"  {label:<50} -> {value}")


def softmax(s, axis=-1):
    """Numerically safe softmax: subtracting the max changes nothing mathematically."""
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention. Q: (..., m, d), K: (..., n, d), V: (..., n, dv).
    mask: boolean (m, n), True where a query MAY look. Returns (output, weights)."""
    d = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    A = softmax(S)
    return A @ V, A


# ---------------------------------------------------------------------------
# 1. A lookup you can differentiate
# ---------------------------------------------------------------------------
print("\n1. A soft dictionary lookup")
# Four stations, each with a 3-number "description" (the key) and a bike count (the value).
names = ["market", "hospital", "stadium", "lakeside"]
keys = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])
values = np.array([7.0, 12.0, 3.0, 9.0])
query = np.array([0.9, 0.1, 0.2])  # "the one by the market", said a bit vaguely
scores = keys @ query
show("dot-product scores (market, hosp, stad, lake)", scores)
show("hard lookup: value of the best key", values[np.argmax(scores)])
sharp_rows = []
for t in [1.0, 4.0, 16.0]:
    w = softmax(t * scores)
    sharp_rows.append(w)
    show(f"soft lookup, sharpness {t:>4}: weights", w)
    show(f"soft lookup, sharpness {t:>4}: answer", round(float(w @ values), 2))

# ---------------------------------------------------------------------------
# 2. Why divide by sqrt(d)
# ---------------------------------------------------------------------------
print("\n2. Scaling: why divide by sqrt(d)")
rng = np.random.default_rng(21)
n_keys = 10
scale_rows = []
for d in [4, 64, 512]:
    q = rng.normal(size=(2000, 1, d))
    k = rng.normal(size=(2000, n_keys, d))
    raw = (q * k).sum(-1)  # (2000, 10)
    out = {}
    for tag, s in [("raw", raw), ("scaled", raw / np.sqrt(d))]:
        a = softmax(s)
        # size of the softmax Jacobian diag(a) - a a^T: how much signal a gradient carries
        jac = np.sqrt(((a[:, :, None] * (np.eye(n_keys) - a[:, None, :])) ** 2).sum((1, 2)))
        out[tag] = (s.std(), a.max(1).mean(), jac.mean())
    scale_rows.append((d, out))
    show(f"d={d:<4} score std raw / scaled", f"{out['raw'][0]:.1f} / {out['scaled'][0]:.2f}")
    show(f"d={d:<4} mean top weight raw / scaled", f"{out['raw'][1]:.3f} / {out['scaled'][1]:.3f}")
    show(f"d={d:<4} softmax gradient size raw / scaled", f"{out['raw'][2]:.4f} / {out['scaled'][2]:.3f}")

# ---------------------------------------------------------------------------
# 3. Self-attention over a sequence, and the causal mask
# ---------------------------------------------------------------------------
print("\n3. Self-attention over a sequence")
n, d_model, d_head = 6, 8, 4
X = rng.normal(size=(n, d_model))  # six reports, eight numbers each
Wq, Wk, Wv = (rng.normal(size=(d_model, d_head)) / np.sqrt(d_model) for _ in range(3))
Y, A = attention(X @ Wq, X @ Wk, X @ Wv)
show("input / output shape", f"{X.shape} / {Y.shape}")
show("each row of weights sums to", A.sum(1))
causal = np.tril(np.ones((n, n), dtype=bool))
Yc, Ac = attention(X @ Wq, X @ Wk, X @ Wv, mask=causal)
show("causal: weight above the diagonal (max)", Ac[~causal].max())
show("causal: report 0 attends to itself with weight", Ac[0, 0])
show("causal: report 5's weights", Ac[5])
perm = rng.permutation(n)
Yp, _ = attention(X[perm] @ Wq, X[perm] @ Wk, X[perm] @ Wv)
show("shuffle the reports: max |out(PX) - P out(X)|", f"{np.abs(Yp - Y[perm]).max():.1e}")

# ---------------------------------------------------------------------------
# 4. Learning the lookup
# ---------------------------------------------------------------------------
print("\n4. Learning to look things up")
D, N_ST, L = 16, 40, 10
station_vec = rng.normal(size=(N_ST, D)) / np.sqrt(D)  # each station's "true" identity
R, _ = np.linalg.qr(rng.normal(size=(D, D)))  # the question arrives in a different language


def make_batch(N, length, rng):
    """Each example: `length` reports from distinct stations (key = noisy identity,
    value = bikes there), and a question about one of them (query = R @ identity + noise)."""
    st = np.array([rng.choice(N_ST, length, replace=False) for _ in range(N)])
    K = station_vec[st] + rng.normal(0, 0.05, (N, length, D))
    V = rng.integers(0, 21, (N, length)).astype(float)
    tgt = rng.integers(0, length, N)
    Q = station_vec[st[np.arange(N), tgt]] @ R.T + rng.normal(0, 0.05, (N, D))
    return Q, K, V, tgt, V[np.arange(N), tgt]


def forward(M, Q, K, V):
    """score_i = q^T M k_i / sqrt(D); answer = sum_i a_i v_i."""
    S = np.einsum("ne,nke->nk", Q @ M, K) / np.sqrt(D)
    A = softmax(S)
    return (A * V).sum(1), A


def loss_and_grad(M, Q, K, V, y):
    out, A = forward(M, Q, K, V)
    g_out = 2 * (out - y) / len(y)  # d(mean squared error)/d(out)
    g_S = A * (V - out[:, None]) * g_out[:, None]  # through the softmax-weighted average
    g_M = Q.T @ np.einsum("nk,nke->ne", g_S, K) / np.sqrt(D)
    return ((out - y) ** 2).mean(), g_M


Qtr, Ktr, Vtr, ttr, ytr = make_batch(3000, L, rng)
Qte, Kte, Vte, tte, yte = make_batch(1000, L, rng)
mae = lambda M: np.abs(forward(M, Qte, Kte, Vte)[0] - yte).mean()

# gradient check on a small slice before trusting the hand-written gradient
M0 = rng.normal(size=(D, D))
_, g = loss_and_grad(M0, Qtr[:50], Ktr[:50], Vtr[:50], ytr[:50])
num = np.zeros(5)
for j, (a, b) in enumerate(zip(range(5), range(3, 8))):
    E = np.zeros((D, D)); E[a, b] = 1e-6
    lp, _ = loss_and_grad(M0 + E, Qtr[:50], Ktr[:50], Vtr[:50], ytr[:50])
    lm, _ = loss_and_grad(M0 - E, Qtr[:50], Ktr[:50], Vtr[:50], ytr[:50])
    num[j] = (lp - lm) / 2e-6
ana = g[np.arange(5), np.arange(3, 8)]
show("gradient check: max relative error", f"{np.abs(num - ana).max() / np.abs(ana).max():.1e}")

# Adam on M, full batch
M = np.zeros((D, D))
m_, v_ = np.zeros_like(M), np.zeros_like(M)
curve = []
for step in range(1, 601):
    l, g = loss_and_grad(M, Qtr, Ktr, Vtr, ytr)
    m_ = 0.9 * m_ + 0.1 * g
    v_ = 0.999 * v_ + 0.001 * g**2
    M -= 0.05 * (m_ / (1 - 0.9**step)) / (np.sqrt(v_ / (1 - 0.999**step)) + 1e-8)
    curve.append(np.sqrt(l))

out_te, A_te = forward(M, Qte, Kte, Vte)
show("MAE: average of all values (no lookup)", f"{np.abs(Vte.mean(1) - yte).mean():.2f} bikes")
show("MAE: attention with M = identity (untrained)", f"{mae(np.eye(D)):.2f} bikes")
show("MAE: attention with learned M", f"{mae(M):.2f} bikes")
hard = Vte[np.arange(len(yte)), np.einsum("ne,nke->nk", Qte @ M, Kte).argmax(1)]
show("MAE: hard argmax with learned M", f"{np.abs(hard - yte).mean():.3f} bikes")
show("weight on the right report, mean (learned)", f"{A_te[np.arange(len(yte)), tte].mean():.3f}")
show("argmax lands on the right report", f"{(A_te.argmax(1) == tte).mean():.1%}")
show("cosine(learned M, the hidden translation R)", f"{(M * R).sum() / np.linalg.norm(M) / np.linalg.norm(R):.3f}")

Q40, K40, V40, t40, y40 = make_batch(1000, 30, rng)
o40, A40 = forward(M, Q40, K40, V40)
show("same M, streams of 30 reports: MAE", f"{np.abs(o40 - y40).mean():.2f} bikes")
show("same M, 30 reports: weight on the right one", f"{A40[np.arange(1000), t40].mean():.3f}")

# ---------------------------------------------------------------------------
# 5. What attention can't see, and what it costs
# ---------------------------------------------------------------------------
print("\n5. Blind to order, and quadratic in length")
q1 = Qte[:1]
o_a, _ = forward(M, q1, Kte[:1], Vte[:1])
p = rng.permutation(L)
o_b, _ = forward(M, q1, Kte[:1, p], Vte[:1, p])
show("true answer for this question", f"{yte[0]:.0f} bikes")
show("answer before / after shuffling the stream", f"{o_a[0]:.4f} / {o_b[0]:.4f}")
for n_tok in [1_000, 8_000, 32_000, 128_000]:
    show(f"score matrix, {n_tok:>7,} tokens, float32, one head", f"{n_tok**2 * 4 / 1e9:,.3g} GB")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for t, w in zip([1, 4, 16], sharp_rows):
    ax[0, 0].plot(names, w, "o-", label=f"sharpness {t}")
ax[0, 0].set_title("1. Soft lookup: weights on each station")
ax[0, 0].legend()
ds = [r[0] for r in scale_rows]
ax[0, 1].plot(ds, [r[1]["raw"][1] for r in scale_rows], "o-", label="raw q.k")
ax[0, 1].plot(ds, [r[1]["scaled"][1] for r in scale_rows], "o-", label="q.k / sqrt(d)")
ax[0, 1].set_xscale("log")
ax[0, 1].set_xlabel("d")
ax[0, 1].set_title("2. Mean top weight over 10 random keys")
ax[0, 1].legend()
im = ax[1, 0].imshow(A_te[:12], cmap="Blues", vmin=0, vmax=1)
ax[1, 0].scatter(tte[:12], np.arange(12), marker="x", color="red", label="right report")
ax[1, 0].set_xlabel("report in the stream")
ax[1, 0].set_ylabel("question")
ax[1, 0].set_title("4. Learned attention weights (12 test questions)")
ax[1, 0].legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax[1, 0])
ax[1, 1].plot(curve)
ax[1, 1].set_xlabel("Adam step")
ax[1, 1].set_ylabel("train RMSE (bikes)")
ax[1, 1].set_title("4. Learning the translation M")
fig.tight_layout()
fig.savefig(OUT / "attention.png", dpi=110)
print(f"\nSaved {OUT.relative_to(HERE)}/attention.png")
