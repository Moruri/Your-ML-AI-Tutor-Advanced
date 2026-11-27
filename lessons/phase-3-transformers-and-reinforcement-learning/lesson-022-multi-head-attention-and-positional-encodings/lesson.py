"""
Lesson 022 - Multi-head attention and positional encodings

Yesterday's dispatcher could answer one question at a time, and couldn't tell
the first report from the last. Today we fix both.

First, multi-head attention: several small attentions side by side, written
as one reshape, checked against doing it head by head. Then a question that
needs two lookups at once ("bikes at A, and at B?"), where one head is stuck
and two heads are not. Then order: a question about *which* report ("what did
the fourth report say?") that attention without positions cannot answer at
all, sinusoidal encodings that fix it, the properties that make them work, and
rotary encodings (RoPE), which make attention depend on distance only.

Run it with (inside your venv):

    python lesson.py

Read it alongside README.md in this folder. Each numbered section here matches
a numbered section there. It saves one figure, `output/heads_and_positions.png`,
next to this script (ignored by git), and runs in well under a minute on a laptop.
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
rng = np.random.default_rng(22)


def show(label, value):
    print(f"  {label:<52} -> {value}")


def softmax(s, axis=-1):
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Lesson 021's scaled dot-product attention, batched over leading axes."""
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    A = softmax(S)
    return A @ V, A


def adam(params, grad_fn, steps, lr):
    """Plain Adam on a list of arrays. grad_fn(params) -> (loss, grads)."""
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    curve = []
    for t in range(1, steps + 1):
        loss, grads = grad_fn(params)
        curve.append(loss)
        for p, g, mi, vi in zip(params, grads, m, v):
            mi *= 0.9; mi += 0.1 * g
            vi *= 0.999; vi += 0.001 * g**2
            p -= lr * (mi / (1 - 0.9**t)) / (np.sqrt(vi / (1 - 0.999**t)) + 1e-8)
    return curve


# ---------------------------------------------------------------------------
# 1. Multi-head attention is a reshape
# ---------------------------------------------------------------------------
print("\n1. Multi-head attention")


def multi_head(X, Wq, Wk, Wv, Wo, h, mask=None):
    """X: (n, d). Project once, split the last axis into h heads of size d/h,
    attend in every head at once, glue the heads back together, mix with Wo."""
    n, d = X.shape
    split = lambda Z: Z.reshape(n, h, d // h).transpose(1, 0, 2)  # (h, n, d/h)
    O, A = attention(split(X @ Wq), split(X @ Wk), split(X @ Wv), mask)
    return O.transpose(1, 0, 2).reshape(n, d) @ Wo, A


n, d = 7, 16
X = rng.normal(size=(n, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
causal = np.tril(np.ones((n, n), dtype=bool))
for h in [1, 2, 4, 8]:
    Y, A = multi_head(X, Wq, Wk, Wv, Wo, h, causal)
    # the slow way: h separate single-head attentions on column slices
    c = d // h
    heads = [attention(X @ Wq[:, i*c:(i+1)*c], X @ Wk[:, i*c:(i+1)*c],
                       X @ Wv[:, i*c:(i+1)*c], causal)[0] for i in range(h)]
    slow = np.concatenate(heads, 1) @ Wo
    show(f"h={h}: weights shape / params / max |fast - slow|",
         f"{A.shape} / {4 * d * d} / {np.abs(Y - slow).max():.1e}")

# ---------------------------------------------------------------------------
# 2. Two questions at once: one head versus two
# ---------------------------------------------------------------------------
print("\n2. Two lookups in one question")
D, N_ST, L = 16, 40, 10
station_vec = rng.normal(size=(N_ST, D)) / np.sqrt(D)
R1, _ = np.linalg.qr(rng.normal(size=(D, D)))
R2, _ = np.linalg.qr(rng.normal(size=(D, D)))


def two_question_batch(N, rng):
    """A stream of L reports (key = noisy station identity, value = bikes) and a
    32-number question that names TWO stations: 'bikes at A? and at B?'."""
    st = np.array([rng.choice(N_ST, L, replace=False) for _ in range(N)])
    K = station_vec[st] + rng.normal(0, 0.05, (N, L, D))
    V = rng.integers(0, 21, (N, L)).astype(float)
    a, b = np.array([rng.choice(L, 2, replace=False) for _ in range(N)]).T
    r = np.arange(N)
    Q = np.concatenate([station_vec[st[r, a]] @ R1.T, station_vec[st[r, b]] @ R2.T], 1)
    Q += rng.normal(0, 0.05, Q.shape)
    return Q, K, V, np.stack([V[r, a], V[r, b]], 1)


def heads_loss(Ms, Q, K, V, Y, which):
    """Each head h: score = q^T M_h k / sqrt(D), output o_h = sum a_i v_i.
    Answer j is read from head which[j]. Returns loss, grads, predictions."""
    outs, As = [], []
    for M in Ms:
        A = softmax(np.einsum("ne,nke->nk", Q @ M, K) / np.sqrt(D))
        outs.append((A * V).sum(1)); As.append(A)
    pred = np.stack([outs[w] for w in which], 1)
    g_pred = 2 * (pred - Y) / Y.size
    grads = []
    for h, (M, A, o) in enumerate(zip(Ms, As, outs)):
        g_o = sum(g_pred[:, j] for j, w in enumerate(which) if w == h)
        g_S = A * (V - o[:, None]) * g_o[:, None]
        grads.append(Q.T @ np.einsum("nk,nke->ne", g_S, K) / np.sqrt(D))
    return ((pred - Y) ** 2).mean(), grads, pred, As


Qtr, Ktr, Vtr, Ytr = two_question_batch(3000, rng)
Qte, Kte, Vte, Yte = two_question_batch(1000, rng)
show("MAE: average of the stream, for both answers", f"{np.abs(Vte.mean(1, keepdims=True) - Yte).mean():.2f} bikes")
results = {}
for name, which in [("one head", [0, 0]), ("two heads", [0, 1])]:
    Ms = [np.zeros((2 * D, D)) for _ in range(max(which) + 1)]
    curve = adam(Ms, lambda P: heads_loss(P, Qtr, Ktr, Vtr, Ytr, which)[:2], 600, 0.05)
    _, _, pred, As = heads_loss(Ms, Qte, Kte, Vte, Yte, which)
    results[name] = (curve, As)
    show(f"MAE: {name}, learned", f"{np.abs(pred - Yte).mean():.2f} bikes")
show("best any single weighting can do: mean |A - B| / 2", f"{np.abs(Yte[:, 0] - Yte[:, 1]).mean() / 2:.2f} bikes")
A1 = results["one head"][1][0]
top2 = np.sort(A1, 1)[:, -2:]
show("one head: mean weight on its top two reports", f"{top2[:, 1].mean():.2f} + {top2[:, 0].mean():.2f}")

# ---------------------------------------------------------------------------
# 3. Order: 'what did report number t say?'
# ---------------------------------------------------------------------------
print("\n3. Which report? Attention needs positions")
DP = 16


def sinusoidal(pos, dim=DP, base=100.0):
    """PE(p)[2i] = sin(p / base^(2i/dim)), PE(p)[2i+1] = cos(same). (The original
    Transformer uses base 10000 for sequences of thousands; ours are short.)"""
    pos = np.asarray(pos, float)[..., None]
    freq = base ** (-np.arange(0, dim, 2) / dim)
    pe = np.zeros(pos.shape[:-1] + (dim,))
    pe[..., 0::2] = np.sin(pos * freq)
    pe[..., 1::2] = np.cos(pos * freq)
    return pe


def position_batch(N, length, ask, rng, use_pe=True):
    """Reports carry content (station identity) and, if use_pe, + PE(position).
    The question is just PE(t): 'what did report t say?'"""
    st = np.array([rng.choice(N_ST, length, replace=False) for _ in range(N)])
    K = station_vec[st] + (sinusoidal(np.arange(length)) if use_pe else 0)
    V = rng.integers(0, 21, (N, length)).astype(float)
    t = rng.choice(ask, N)
    return sinusoidal(t), K, V, t, V[np.arange(N), t]


def one_head(M, Q, K, V, y):
    loss, grads, pred, As = heads_loss([M], Q, K, V, y[:, None], [0])
    return loss, grads, pred[:, 0], As[0]


L2 = 12
for use_pe in [False, True]:
    Qtr, Ktr, Vtr, ttr, ytr = position_batch(3000, L2, np.arange(L2), rng, use_pe)
    Qte, Kte, Vte, tte, yte = position_batch(1000, L2, np.arange(L2), rng, use_pe)
    M = np.zeros((DP, DP))
    curve = adam([M], lambda P: one_head(P[0], Qtr, Ktr, Vtr, ytr)[:2], 600, 0.05)
    _, _, pred, A = one_head(M, Qte, Kte, Vte, yte)
    tag = "with sinusoidal positions" if use_pe else "no positions"
    show(f"MAE: {tag}", f"{np.abs(pred - yte).mean():.2f} bikes")
    show(f"weight on report t: {tag}", f"{A[np.arange(1000), tte].mean():.3f}")
M_pos, A_pos, t_pos = M, A, tte
show("MAE: average of the stream", f"{np.abs(Vte.mean(1) - yte).mean():.2f} bikes")
Q24, K24, V24, t24, y24 = position_batch(1000, 24, np.arange(12, 24), rng)
_, _, p24, A24 = one_head(M_pos, Q24, K24, V24, y24)
show("same M, streams of 24, asking about reports 12-23", f"MAE {np.abs(p24 - y24).mean():.2f} bikes, "
     f"weight on t {A24[np.arange(1000), t24].mean():.3f}")

# ---------------------------------------------------------------------------
# 4. What makes sinusoids a good position code
# ---------------------------------------------------------------------------
print("\n4. Properties of the sinusoidal code")
P = sinusoidal(np.arange(200))
show("norm of every PE(p) (min / max)", f"{np.linalg.norm(P, axis=1).min():.4f} / {np.linalg.norm(P, axis=1).max():.4f}")
show("PE(3).PE(7), PE(50).PE(54), PE(120).PE(124)", f"{P[3] @ P[7]:.4f}, {P[50] @ P[54]:.4f}, {P[120] @ P[124]:.4f}")
show("PE(p).PE(p+k) for k = 0, 1, 2, 5, 10, 40", np.array([P[60] @ P[60 + k] for k in [0, 1, 2, 5, 10, 40]]))


def shift_matrix(k, dim=DP, base=100.0):
    """A block-diagonal rotation: PE(p + k) = T_k @ PE(p), for every p."""
    T = np.zeros((dim, dim))
    for i, f in enumerate(base ** (-np.arange(0, dim, 2) / dim)):
        c, s = np.cos(k * f), np.sin(k * f)
        T[2*i:2*i+2, 2*i:2*i+2] = [[c, s], [-s, c]]
    return T


T5 = shift_matrix(5)
show("max |PE(p + 5) - T_5 PE(p)| over p = 0..194", f"{np.abs(P[5:] - P[:-5] @ T5.T).max():.1e}")

# ---------------------------------------------------------------------------
# 5. Rotary encodings: make the score depend on distance only
# ---------------------------------------------------------------------------
print("\n5. Rotary position encodings (RoPE)")


def rope(x, pos, base=100.0):
    """Rotate each pair (x[2i], x[2i+1]) by angle pos * freq_i."""
    dim = x.shape[-1]
    ang = np.asarray(pos, float)[..., None] * base ** (-np.arange(0, dim, 2) / dim)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out


q, k = rng.normal(size=DP), rng.normal(size=DP)
s_rope = [rope(q, m) @ rope(k, m - 3) for m in [3, 10, 100]]
s_abs = [(q + sinusoidal(m)) @ (k + sinusoidal(m - 3)) for m in [3, 10, 100]]
show("RoPE score, query at m, key at m-3, m=3/10/100", np.array(s_rope))
show("added-PE score, same pairs", np.array(s_abs))
show("RoPE keeps lengths: |rope(q, 37)| - |q|", f"{np.linalg.norm(rope(q, 37)) - np.linalg.norm(q):.1e}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for name, (curve, _) in results.items():
    ax[0, 0].plot(np.sqrt(curve), label=name)
ax[0, 0].set_xlabel("Adam step"); ax[0, 0].set_ylabel("train RMSE (bikes)")
ax[0, 0].set_title("2. Two lookups: one head stalls, two heads don't")
ax[0, 0].legend()
As2 = results["two heads"][1]
show_n = 6
img = np.concatenate([As2[0][:show_n], np.full((show_n, 1), np.nan), As2[1][:show_n]], 1)
ax[0, 1].imshow(img, cmap="Blues", vmin=0, vmax=1)
ax[0, 1].set_title("2. Two heads, 6 questions: head 1 | head 2")
ax[0, 1].set_xlabel("report in the stream"); ax[0, 1].set_ylabel("question")
im = ax[1, 0].imshow(P[:60].T, aspect="auto", cmap="RdBu")
ax[1, 0].set_xlabel("position"); ax[1, 0].set_ylabel("dimension")
ax[1, 0].set_title("3-4. Sinusoidal codes: fast to slow waves")
fig.colorbar(im, ax=ax[1, 0])
ax[1, 1].plot(np.arange(60), P[20] @ P[:60].T)
ax[1, 1].axvline(20, color="grey", ls=":")
ax[1, 1].set_xlabel("position p"); ax[1, 1].set_ylabel("PE(20) . PE(p)")
ax[1, 1].set_title("4. Similarity peaks at the same position")
fig.tight_layout()
fig.savefig(OUT / "heads_and_positions.png", dpi=110)
print(f"\nSaved {OUT.relative_to(HERE)}/heads_and_positions.png")
