"""Hand-check reference formulas for the Run E mission simulations.

Every function here is derived from first principles and written without
looking at the mission simulator, so tests can compare the simulator against
an independent answer.  All results are for SIMULATED scenarios only.  Radio
numbers used with these formulas are literature-derived, never measured on
Zephyr hardware.

Units: seconds, metres, metres per second, Hz, dB and dBm throughout.

Shared conventions
------------------
Heartbeat grid: heartbeats are sent at t = k / h (h in Hz).  Radio delay is
ignored, so a heartbeat that arrives arrives at its send time.  The lost-link
alarm fires when the time since the newest received heartbeat becomes
strictly greater than the timeout T.  A gap of exactly T does NOT alarm.
"""
from __future__ import annotations

import math

import numpy as np

# Small tolerance used when a product like T * h should be an integer but
# floating point lands a hair below it (for example 0.7 * 10).
_GRID_EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. Straight-line tail follow
# ---------------------------------------------------------------------------
def tail_follow_time_s(initial_gap_m, zone_radius_m, follower_speed, target_speed):
    """Time for a follower directly behind the target to reach the tag zone.

    Derivation: both move on the same straight line, the target straight
    away from the follower.  The gap shrinks at the closing speed
    (v_f - v_t).  The follower is in the zone once the gap is down to the
    zone radius, so it must close (gap - radius) metres:

        t = (gap - radius) / (v_f - v_t)

    If the follower is not faster (v_f <= v_t) the gap never shrinks and the
    time is infinite.  If it already starts inside the zone the time is 0.

    Named "tail follow" because the project's word list keeps other names
    for this geometry out of the code.
    """
    gap = float(initial_gap_m)
    radius = float(zone_radius_m)
    if gap <= radius:
        return 0.0
    closing = float(follower_speed) - float(target_speed)
    if closing <= 0.0:
        return math.inf
    return (gap - radius) / closing


# ---------------------------------------------------------------------------
# 2. Heartbeat false alarms
# ---------------------------------------------------------------------------
def misses_needed(rate_hz, timeout_s):
    """Consecutive lost heartbeats needed before the alarm fires.

    If heartbeat k arrives and the next arrival is heartbeat k + g, the gap
    is g / h seconds and there are g - 1 misses in between.  The alarm needs
    g / h > T, so g > T h.  The smallest such g is floor(T h) + 1 whether or
    not T h is a whole number, so the misses needed are

        m = floor(T h)

    (a gap of exactly T, g = T h, does not alarm).  A tiny tolerance stops
    floating point from turning 7.0 into 6.999...
    """
    return int(math.floor(float(timeout_s) * float(rate_hz) + _GRID_EPS))


def heartbeat_false_alarms_per_hour_iid(p, rate_hz, timeout_s):
    """Expected false alarms per hour when every heartbeat is lost independently.

    Derivation: one false alarm per run of at least m consecutive losses.
    Count each run by its starting slot: heartbeat k arrives (chance 1 - p)
    and heartbeats k + 1 ... k + m are all lost (chance p**m).  Runs longer
    than m still count once because only the slot right after an arrival can
    start a run.  There are 3600 h slots per hour, so

        rate = 3600 h (1 - p) p**m,  m = floor(T h).

    p = 0 gives 0.  The slow-start edge (the hour starting mid-run) is ignored.
    """
    p = float(p)
    h = float(rate_hz)
    m = misses_needed(h, timeout_s)
    return 3600.0 * h * (1.0 - p) * p ** m


def false_alarms_per_hour_markov(p_enter_bad, p_exit_bad, rate_hz, timeout_s):
    """Exact false alarms per hour for a two-state burst chain on the heartbeat grid.

    Model: one chain step per heartbeat slot.  In the good state every
    heartbeat arrives; in the bad state every heartbeat is lost.  From good
    the chain enters bad with chance q = p_enter_bad per slot; from bad it
    leaves with chance r = p_exit_bad per slot.

    Derivation: an alarm needs m = floor(T h) consecutive bad slots.  Count
    each bad run by its first slot: the slot before is good (stationary
    chance pi_good = r / (q + r)), the chain enters bad (q), and then stays
    bad for m - 1 more slots ((1 - r)**(m - 1)).  Per hour:

        rate = 3600 h pi_good q (1 - r)**(m - 1).

    With Run C's slot convention (r = 1 / (D h), q = p r / (1 - p)) one has
    pi_good = 1 - p and pi_good q = p r, which gives Run C's published form
    3600 h p r (1 - r)**(m - 1).  Needs m >= 1 (T at least one slot).
    """
    q = float(p_enter_bad)
    r = float(p_exit_bad)
    h = float(rate_hz)
    m = misses_needed(h, timeout_s)
    if m < 1:
        raise ValueError("timeout must be at least one heartbeat period")
    if q == 0.0:
        return 0.0
    if q + r == 0.0:
        raise ValueError("chain with q = r = 0 has no unique stationary state")
    pi_good = r / (q + r)
    return 3600.0 * h * pi_good * q * (1.0 - r) ** (m - 1)


def per_slot_chain(mean_loss, mean_burst_s, rate_hz):
    """Run C's slot convention for a burst chain: returns (q, r).

    Mean burst length D seconds is D h slots, and a geometric burst with exit
    chance r per slot has mean 1 / r slots, so r = 1 / (D h).  The long-run
    loss is pi_bad = q / (q + r); setting it to p gives q = p r / (1 - p).
    """
    p = float(mean_loss)
    r = 1.0 / (float(mean_burst_s) * float(rate_hz))
    if not 0.0 < r <= 1.0:
        raise ValueError("mean burst must be at least one slot")
    q = p * r / (1.0 - p)
    return q, r


def coarse_grid_chain(q0, r0, k):
    """Chain seen when a fine-grid burst chain is sampled every k steps.

    The fine chain runs on the 0.01 s control grid with entry chance q0 and
    exit chance r0 per step.  A heartbeat sent every k steps (20 Hz -> k = 5,
    10 Hz -> k = 10) only sees the state at every k-th step.

    Sampling a Markov chain every k steps is again a Markov chain: given the
    state now, the state k steps later depends only on the present, and its
    transition matrix is the k-th power of the one-step matrix

        P = [[1 - q0, q0],
             [r0,     1 - r0]].

    So q = (P**k)[good, bad] and r = (P**k)[bad, good].

    What is exact and what is not: the joint pattern of lost and delivered
    heartbeats is reproduced exactly, so false alarm counts from
    false_alarms_per_hour_markov(q, r, ...) are exact for heartbeat loss.
    Burst identity is not: two samples that are both bad may belong to two
    different fine bursts with a short good spell in between, which the
    coarse chain treats as one burst.  The test suite measures how often that
    happens against a brute-force fine-chain simulation.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive whole number of fine steps")
    P = np.array([[1.0 - q0, q0], [r0, 1.0 - r0]], dtype=float)
    Pk = np.linalg.matrix_power(P, k)
    return float(Pk[0, 1]), float(Pk[1, 0])


# ---------------------------------------------------------------------------
# 3. Detection delay of a permanent link failure
# ---------------------------------------------------------------------------
def detection_delay_bounds(rate_hz, timeout_s):
    """(min, max, mean) delay from a permanent failure to the alarm.

    Convention: a heartbeat sent at the exact failure instant still gets
    through (the failure takes effect just after it), and the failure time is
    uniform within a heartbeat period.

    Derivation: the newest heartbeat that got through was sent at t_last, with
    t_fail - 1/h < t_last <= t_fail.  The alarm fires at t_last + T, so the
    delay t_last + T - t_fail lies in (T - 1/h, T].  It is uniform there, so
    the mean is T - 1/(2h).  The minimum T - 1/h is a bound, never reached.
    Needs T >= 1/h, otherwise the alarm fires between healthy heartbeats.
    """
    h = float(rate_hz)
    T = float(timeout_s)
    if T * h < 1.0 - _GRID_EPS:
        raise ValueError("timeout shorter than one heartbeat period")
    period = 1.0 / h
    return (T - period, T, T - period / 2.0)


# ---------------------------------------------------------------------------
# 4. Stopping buffer
# ---------------------------------------------------------------------------
def stopping_buffer_m(speed, data_age_s, reaction_s, max_decel):
    """Distance covered while acting on old data and then braking to a stop.

    Derivation: for (data age + reaction time) the vehicle keeps its speed v
    because it has not yet learned it must stop: v (age + reaction).  Then it
    brakes at constant deceleration a, which takes v / a seconds and covers
    v**2 / (2 a) (area of the speed-time triangle).  Total:

        buffer = v (age + reaction) + v**2 / (2 a).

    Run E uses this as its stand-in separation buffer because Run D's buffer
    is not available.
    """
    v = float(speed)
    a = float(max_decel)
    if a <= 0.0:
        raise ValueError("max_decel must be positive")
    return v * (float(data_age_s) + float(reaction_s)) + v * v / (2.0 * a)


# ---------------------------------------------------------------------------
# 5. Area coverage
# ---------------------------------------------------------------------------
def coverage_time_s(area_m2, n_drones, swath_m, speed):
    """Ideal time for N drones to sweep an area once.

    Derivation: one drone flying at v with sensor swath w covers w v square
    metres per second.  N drones with no overlap and no time lost in turns
    cover N w v per second, so the time is A / (N w v).  Real sweeps take
    longer; this is a lower bound.
    """
    rate = float(n_drones) * float(swath_m) * float(speed)
    if rate <= 0.0:
        return math.inf
    return float(area_m2) / rate


# ---------------------------------------------------------------------------
# 6. Path loss, range and relay geometry
# ---------------------------------------------------------------------------
def log_distance_path_loss_db(d_m, pl0_db, exponent, d0=1.0):
    """Log-distance path loss: PL0 + 10 n log10(d / d0).

    Received power falls as 1 / d**n, so in dB the loss grows by 10 n dB per
    factor of ten in distance, starting from PL0 at the reference distance d0.
    """
    return float(pl0_db) + 10.0 * float(exponent) * math.log10(float(d_m) / float(d0))


def dual_slope_path_loss_db(d_m, pl0_db, n1, breakpoint_m, n2, d0=1.0):
    """Two-slope path loss: exponent n1 up to the breakpoint, n2 beyond it.

    Continuous at the breakpoint:
        d <= b: PL0 + 10 n1 log10(d / d0)
        d >  b: PL0 + 10 n1 log10(b / d0) + 10 n2 log10(d / b)
    """
    d = float(d_m)
    b = float(breakpoint_m)
    if d <= b:
        return log_distance_path_loss_db(d, pl0_db, n1, d0)
    return log_distance_path_loss_db(b, pl0_db, n1, d0) + 10.0 * float(n2) * math.log10(d / b)


def max_range_m(tx_dbm, sensitivity_dbm, pl0_db, exponent, margin_db, d0=1.0):
    """Largest distance where received power still clears sensitivity plus a margin.

    Set tx - PL(d) = sensitivity + margin and solve the log-distance model
    for d:
        d = d0 * 10 ** ((tx - sensitivity - margin - PL0) / (10 n)).
    No shadowing or fading: this is the median-signal range.
    """
    allowed_loss = float(tx_dbm) - float(sensitivity_dbm) - float(margin_db)
    return float(d0) * 10.0 ** ((allowed_loss - float(pl0_db)) / (10.0 * float(exponent)))


def max_range_dual_slope_m(tx_dbm, sensitivity_dbm, pl0_db, n1, breakpoint_m, n2, margin_db, d0=1.0):
    """Same as max_range_m but for the two-slope model.

    If the allowed loss runs out before the breakpoint the single-slope answer
    stands; otherwise the remaining dB beyond the breakpoint are spent at
    10 n2 dB per decade: d = b * 10 ** ((allowed - PL(b)) / (10 n2)).
    """
    allowed_loss = float(tx_dbm) - float(sensitivity_dbm) - float(margin_db)
    b = float(breakpoint_m)
    loss_at_b = log_distance_path_loss_db(b, pl0_db, n1, d0)
    if allowed_loss <= loss_at_b:
        return max_range_m(tx_dbm, sensitivity_dbm, pl0_db, n1, margin_db, d0)
    return b * 10.0 ** ((allowed_loss - loss_at_b) / (10.0 * float(n2)))


def relay_feasible(total_distance_m, relay_fraction, max_range):
    """True if a relay placed relay_fraction of the way along a straight line closes both hops.

    Hop 1 is f D long, hop 2 is (1 - f) D.  Each hop must be no longer than
    the single-link range (at exactly max range the link meets its margin
    exactly, which counts as closing).
    """
    D = float(total_distance_m)
    f = float(relay_fraction)
    if not 0.0 <= f <= 1.0:
        raise ValueError("relay_fraction must be in [0, 1]")
    return f * D <= float(max_range) and (1.0 - f) * D <= float(max_range)


def best_relay_fraction():
    """Relay position that allows the longest total distance for two equal radios.

    The link fails at its longer hop, max(f, 1 - f) D.  That is smallest at
    f = 0.5, where both hops are D / 2, so a midpoint relay doubles the
    reachable distance to 2 x max range.
    """
    return 0.5


# ---------------------------------------------------------------------------
# 7. Alert latency
# ---------------------------------------------------------------------------
def alert_latency_s(detection_s, network_delay_s, processing_s):
    """End-to-end alert latency: detection + network + processing, in series."""
    return float(detection_s) + float(network_delay_s) + float(processing_s)


def expected_detection_s(fix_period_s, k_consecutive):
    """Mean time from zone entry to the k-th consecutive in-zone fix.

    Assumptions: fixes every Ts seconds, the entry time is uniform between
    fixes, the object stays in the zone, and every fix is correct and
    delivered (no noise, no loss).

    Derivation: the first fix after entry comes U Ts later with U uniform on
    (0, 1], mean Ts / 2.  The alert needs k consecutive in-zone fixes, so it
    waits k - 1 more periods.  Delay = U Ts + (k - 1) Ts, mean (k - 0.5) Ts,
    range ((k - 1) Ts, k Ts].
    """
    k = int(k_consecutive)
    if k < 1:
        raise ValueError("k_consecutive must be at least 1")
    return (k - 0.5) * float(fix_period_s)


# ---------------------------------------------------------------------------
# 8. Position jitter of a first-order loop closed on noisy fixes
# ---------------------------------------------------------------------------
def first_order_loop_jitter_std(sigma_m, fix_period_s, response_time_s):
    """Steady-state standard deviation of the true position, one axis.

    Model and assumptions:
    - the vehicle's position follows its setpoint with a first-order response
      of time constant tau: dx/dt = (s - x) / tau;
    - every Ts a fix z_k = x_k + n_k arrives, with n_k white, zero mean,
      standard deviation sigma, independent of everything else;
    - the loop sets s_k = x_k + (g - z_k) = g - n_k (move by the measured
      error) and holds it until the next fix;
    - no wind or other disturbance.

    Over one hold, x relaxes a fraction a = 1 - exp(-Ts / tau) of the way to
    s_k.  With e = x - g:
        e_{k+1} = e_k + a (-n_k - e_k) = (1 - a) e_k - a n_k.
    In steady state var(e) = (1 - a)**2 var(e) + a**2 sigma**2, so
        var = a**2 sigma**2 / (1 - (1 - a)**2) = a sigma**2 / (2 - a).
    This is the spread at the fix instants.  For Ts much smaller than tau,
    a is about Ts / tau and std is about sigma sqrt(Ts / (2 tau)).
    """
    sigma = float(sigma_m)
    a = 1.0 - math.exp(-float(fix_period_s) / float(response_time_s))
    return sigma * math.sqrt(a / (2.0 - a))


# ---------------------------------------------------------------------------
# 9. Lag from late setpoints
# ---------------------------------------------------------------------------
def lag_error_m(speed, delay_s):
    """Along-track lag from acting on setpoints delay seconds old: v d.

    A goal moving at constant speed v was v d metres back when the setpoint
    was made, so a late setpoint adds only that lag (Run A arm a).
    """
    return float(speed) * float(delay_s)
