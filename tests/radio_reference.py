"""Small standalone reference simulator for random-access radio checks.

Written from the probability alone (no code from sim/radio is used or read).
It exists so the MAC engine can be checked against something that shares none
of its code.  Three models:

* Pure ALOHA: frames of fixed duration T start at Poisson times.  A frame
  succeeds iff no other frame starts within T of it (either side).  Sort the
  start times; only the two neighbours matter, so a frame succeeds iff the gap
  to both neighbours is at least T.  Closed form: S = G exp(-2G).
* Slotted ALOHA: frames start on slot boundaries (slot length T).  A frame
  succeeds iff it is the only frame in its slot.  Closed form: S = G exp(-G).
* Single-sender FIFO queue with Poisson arrivals and deterministic service D
  (M/D/1).  Waiting times follow the Lindley recursion
  W_{n+1} = max(0, W_n + D - A_{n+1}).  Closed form (Pollaczek-Khinchine):
  Wq = rho / (2 mu (1 - rho)) with mu = 1 / D and rho = lambda D.

G = lambda T is the offered load: the mean number of frame starts per frame
duration.  S = G * P(success) is the mean number of successes per frame
duration.

Also here: a finite-population variant of ALOHA (N nodes, each half-duplex, so
a node that gets a new arrival while its own previous frame is still on air
must wait until that frame ends).  This is the exact situation the MAC engine
runs, so it gives the right number to compare the engine against (the
infinite-population closed form is then a limit as N grows).

Everything is deterministic per integer seed.
"""

from __future__ import annotations

import numpy as np

# Two frames whose start times differ by T times (1 - EPS) or more are treated
# as not overlapping (back-to-back frames of one node differ by exactly T, up to
# floating point rounding).
EPS = 1e-9


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


# ----------------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------------

def pure_aloha_success_prob(G: float) -> float:
    """P(a frame succeeds) = exp(-2G): no other start in a window of length 2T."""
    return float(np.exp(-2.0 * G))


def slotted_aloha_success_prob(G: float) -> float:
    """P(a frame succeeds) = exp(-G): no other frame in its slot."""
    return float(np.exp(-G))


def md1_mean_wait(lam: float, D: float) -> float:
    """Pollaczek-Khinchine mean wait in queue for M/D/1: lam D^2 / (2 (1 - rho))."""
    rho = lam * D
    if not 0.0 <= rho < 1.0:
        raise ValueError("need 0 <= rho < 1")
    return lam * D * D / (2.0 * (1.0 - rho))


# ----------------------------------------------------------------------------
# Sampling error helpers
# ----------------------------------------------------------------------------

def batch_means(x, n_batches: int = 20) -> tuple[float, float]:
    """Mean of x and its standard error from n_batches equal consecutive batches.

    Any left-over samples at the end are dropped.  Batch means are close to
    independent when each batch is much longer than the correlation time, so
    the standard error is std(batch means, ddof=1) / sqrt(n_batches).
    """
    x = np.asarray(x, dtype=float)
    size = len(x) // n_batches
    if size < 1:
        raise ValueError("too few samples for the number of batches")
    b = x[: size * n_batches].reshape(n_batches, size).mean(axis=1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(n_batches))


def pure_aloha_fraction_se(G: float, n: int) -> float:
    """Standard error of the success fraction over n consecutive frames (pure ALOHA).

    Success indicator I_i = [gap before >= T][gap after >= T] with iid
    exponential gaps, q = exp(-G) = P(gap >= T).  Then p = q^2 and, because
    neighbouring frames share one gap, Cov(I_i, I_{i+1}) = q^3 - q^4; frames two
    or more apart are independent.  So
        Var(mean) = [p (1 - p) + 2 (q^3 - q^4)] / n   (large n).
    This is larger than the plain binomial value p (1 - p) / n.
    """
    q = np.exp(-G)
    p = q * q
    var = (p * (1.0 - p) + 2.0 * (q ** 3 - q ** 4)) / n
    return float(np.sqrt(var))


# ----------------------------------------------------------------------------
# Classification of a set of start times (used on any list of starts)
# ----------------------------------------------------------------------------

def classify_pure(starts, T: float, eps: float = EPS) -> np.ndarray:
    """Per-frame success for pure ALOHA given frame start times (any order).

    A frame fails iff another frame starts strictly less than T away from it.
    Returned array is in the order of the input.
    """
    s = np.asarray(starts, dtype=float)
    order = np.argsort(s, kind="stable")
    ss = s[order]
    ok = np.ones(len(ss), dtype=bool)
    if len(ss) > 1:
        close = np.diff(ss) < T * (1.0 - eps)  # gap i is between sorted frame i and i+1
        ok[:-1] &= ~close
        ok[1:] &= ~close
    out = np.empty(len(ss), dtype=bool)
    out[order] = ok
    return out


def classify_slotted(starts, T: float, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame success for slotted ALOHA given frame start times.

    Every start must sit on a slot boundary (multiple of T within tol slots).
    Returns (success, slot_index); a frame succeeds iff alone in its slot.
    """
    s = np.asarray(starts, dtype=float)
    u = s / T
    slot = np.rint(u).astype(np.int64)
    if len(s) and np.max(np.abs(u - slot)) > tol:
        raise ValueError("start times are not aligned to slot boundaries")
    _, inv, cnt = np.unique(slot, return_inverse=True, return_counts=True)
    return cnt[inv] == 1, slot


# ----------------------------------------------------------------------------
# Infinite-population simulators (Poisson stream of attempts, no queueing)
# ----------------------------------------------------------------------------

def simulate_pure_aloha(G: float, n_frames: int, rng: np.random.Generator, T: float = 1.0) -> dict:
    """Pure ALOHA with an overall Poisson stream of rate G / T.

    n_frames interior frames are scored (one extra on each side supplies
    neighbours).  Returns success_fraction, throughput S = G * fraction, and
    their standard errors (se includes the neighbour correlation).
    """
    if G <= 0 or n_frames < 2:
        raise ValueError("need G > 0 and n_frames >= 2")
    gaps = rng.exponential(T / G, size=n_frames + 2)
    d = gaps[1:]  # d[k] = start[k+1] - start[k]; frame k+1 (k = 0..n-1) has gaps d[k], d[k+1]
    ok = (d[:-1] >= T) & (d[1:] >= T)
    frac = float(ok.mean())
    se = pure_aloha_fraction_se(G, n_frames)
    return {"success_fraction": frac, "throughput": G * frac, "se_fraction": se,
            "se_throughput": G * se, "n": int(len(ok))}


def simulate_slotted_aloha(G: float, n_frames: int, rng: np.random.Generator) -> dict:
    """Slotted ALOHA: the number of frames per slot is Poisson(G), iid across slots.

    About n_frames frames are generated (n_slots = ceil(n_frames / G)).
    throughput S = (slots holding exactly one frame) / n_slots is an exact mean of
    iid Bernoulli(G exp(-G)) values, so se_throughput is the binomial value.
    The per-frame success fraction is a ratio of two sums; its standard error is
    the first-order (delta method) value from the empirical moments.
    """
    if G <= 0 or n_frames < 2:
        raise ValueError("need G > 0 and n_frames >= 2")
    n_slots = int(np.ceil(n_frames / G))
    c = rng.poisson(G, size=n_slots)
    x = (c == 1).astype(float)
    S = float(x.mean())
    se_S = float(np.sqrt(S * (1.0 - S) / n_slots))
    total = float(c.sum())
    f = float(x.sum() / total)
    ybar = total / n_slots
    resid = x - f * c  # linearised residual of the ratio estimator
    se_f = float(np.sqrt(resid.var(ddof=1) / n_slots) / ybar)
    return {"success_fraction": f, "throughput": S, "se_fraction": se_f,
            "se_throughput": se_S, "n": int(total), "n_slots": n_slots}


# ----------------------------------------------------------------------------
# Finite population: N half-duplex nodes, per-node FIFO deferral
# ----------------------------------------------------------------------------

def defer_pure(arrivals, T: float) -> np.ndarray:
    """Start times of one node's frames: start_k = max(arrival_k, start_{k-1} + T).

    Unrolled: start_k = k T + max_{j <= k} (arrival_j - j T).  arrivals sorted.
    """
    a = np.asarray(arrivals, dtype=float)
    k = np.arange(len(a))
    return k * T + np.maximum.accumulate(a - k * T)


def defer_slotted(arrivals, T: float) -> np.ndarray:
    """Slot-aligned start times of one node's frames.

    Slot index m_k = max(ceil(arrival_k / T), m_{k-1} + 1): a frame goes in the next
    slot boundary at or after its arrival, and one slot after the node's previous
    frame at the earliest.  Unrolled: m_k = k + max_{j <= k} (ceil(a_j / T) - j).
    """
    a = np.asarray(arrivals, dtype=float)
    k = np.arange(len(a))
    c = np.ceil(a / T)
    return (k + np.maximum.accumulate(c - k)) * T


def finite_n_aloha(G: float, n_nodes: int, t_end: float, rng: np.random.Generator, *,
                   T: float = 1.0, slotted: bool = False) -> dict:
    """N nodes, each a Poisson source of rate G / (T N) on [0, t_end], half-duplex.

    Returns arrays sorted by creation time: created, start, node, success.
    success uses all frames (including those deferred past t_end) as neighbours.
    """
    if G <= 0 or n_nodes < 1 or t_end <= 0:
        raise ValueError("bad arguments")
    lam = G / (T * n_nodes)
    counts = rng.poisson(lam * t_end, size=n_nodes)
    created, start, node = [], [], []
    for i, c in enumerate(counts):
        a = np.sort(rng.uniform(0.0, t_end, size=int(c)))
        s = defer_slotted(a, T) if slotted else defer_pure(a, T)
        created.append(a)
        start.append(s)
        node.append(np.full(int(c), i, dtype=np.int64))
    created = np.concatenate(created)
    start = np.concatenate(start)
    node = np.concatenate(node)
    order = np.argsort(created, kind="stable")
    created, start, node = created[order], start[order], node[order]
    if slotted:
        success, _ = classify_slotted(start, T)
    else:
        success = classify_pure(start, T)
    return {"created": created, "start": start, "node": node, "success": success}


def window_success(res: dict, t0: float, t1: float, n_batches: int = 20) -> tuple[float, float, int]:
    """(mean success, batch-means se, n frames) over frames created in [t0, t1)."""
    m = (res["created"] >= t0) & (res["created"] < t1)
    x = res["success"][m].astype(float)
    mean, se = batch_means(x, n_batches)
    return mean, se, int(m.sum())


# ----------------------------------------------------------------------------
# Single sender with Poisson arrivals and deterministic service (M/D/1)
# ----------------------------------------------------------------------------

def md1_waits(lam: float, D: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Waiting times W_1..W_n in queue (arrival to start of service), M/D/1.

    Lindley: W_{k+1} = max(0, W_k + D - A_{k+1}) with W_1 = 0 (first arrival finds
    the system empty).  Unrolled with S_k = sum_{j<=k} (D - A_{j+1}), S_0 = 0:
    W_k = S_k - min(S_0, ..., S_k).
    """
    if lam <= 0 or D <= 0 or n < 1:
        raise ValueError("bad arguments")
    a = rng.exponential(1.0 / lam, size=n)
    s = np.concatenate(([0.0], np.cumsum(D - a[1:])))
    return s - np.minimum.accumulate(s)


def single_sender_starts(arrivals, T: float, difs: float) -> np.ndarray:
    """Transmission start times of one carrier-sensing sender with no backoff.

    Rule (documented in sim/radio/mac.py): the medium is otherwise idle.  A frame becomes
    head of line on arrival, or when the previous frame ends at e_{k-1}; it then
    always waits DIFS of idle medium before starting:
        start_k = max(arrival_k, start_{k-1} + T) + difs
    With s_k = start_k - difs (the moment service begins) and D = T + difs:
        s_k = max(arrival_k, s_{k-1} + D)
    which is the Lindley recursion with service time D.  Unrolled:
    start_k = difs + k D + max_{j <= k} (arrival_j - j D).
    """
    a = np.asarray(arrivals, dtype=float)
    D = T + difs
    k = np.arange(len(a))
    return difs + k * D + np.maximum.accumulate(a - k * D)


def head_of_line_waits(arrivals, starts, T: float) -> np.ndarray:
    """Time from arrival until the frame becomes head of line.

    Frame k becomes head of line when frame k-1 has finished transmitting, at
    end_{k-1} = start_{k-1} + T, or on arrival if the queue was empty.  So
    H_k = max(0, end_{k-1} - arrival_k), H_1 = 0.  (Algebra: H = W - difs where
    W = start - arrival, because every frame waits DIFS after head of line; H is
    the Lindley waiting time, i.e. the M/D/1 queueing delay.)
    """
    a = np.asarray(arrivals, dtype=float)
    s = np.asarray(starts, dtype=float)
    h = np.zeros(len(a))
    h[1:] = np.maximum(0.0, s[:-1] + T - a[1:])
    return h
