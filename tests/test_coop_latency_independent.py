"""Independent checks on how link delay and packet loss reach the cooperative estimate.

Setup: one drone watches one target that moves in a straight line at 0.8 m/s.
The drone's camera has zero noise. A ground station fuses the drone's
measurements and sends the result back. The drone then carries that old
result forward to "now". Nothing here is copied from the simulator: the
expected numbers come from the small helpers below, written from the
textbook equations.

Why the age is what it is
-------------------------
Information makes a round trip. The drone measures on a 0.1 s clock. The
measurement takes 0.17 s to reach the ground station. The station only sends
a summary on its own clock (every 0.5 s in Run A), and it uses the newest
measurement it has at that moment. The summary takes another 0.17 s to come
back, and the drone then keeps using the same summary until a newer one
arrives. So the "age" of the information is delay in, plus waiting for the
clocks to line up, plus delay out, plus however long the drone has been
holding the same summary. With no loss that gives ages of 0.40, 0.45, ...,
0.85 s, repeating every 0.5 s.

With loss, every lost summary pushes the usable one back one share period,
and every lost measurement pushes the newest measurement back one capture
period. The number of losses in a row before the first success is a
"geometric" count: it is 0 with chance (1 - p), 1 with chance p (1 - p),
2 with chance p^2 (1 - p), and so on. So the age is the minimum age plus 0.1 s
times the sum of two such counts (one per link).

Why the position variance grows like P_pp + 2 P_pv a + P_vv a^2 + q a^3 / 3
---------------------------------------------------------------------------
Suppose the station knows the position with some error, and knows the speed
with some error. After a seconds of coasting, the position error is the old
position error plus (speed error) times a. Squaring and averaging that gives
P_pp (old position uncertainty), 2 P_pv a (the two old errors can help or
fight each other), and P_vv a^2 (speed uncertainty times time, squared).
On top of that the target can get random pushes while we coast. A push at
time tau changes the position by (a - tau) times the push, and independent
pushes add up in squares, so the total is q times the area under
(a - tau)^2, which is q a^3 / 3. The covariance trace is three times this
(the same thing happens on the x, y and z axes).

Why the third difference is 6 q h^3
-----------------------------------
While the drone holds one summary its age goes up by h = 0.05 s each step,
so the trace is a cubic in the age: constant + linear + quadratic + q a^3.
Taking "next minus this" once drops a power; doing it three times kills the
constant, the line, and the parabola completely (like the third derivative
of a cubic being a constant) and leaves only the cubic term. For c a^3 the
third difference is 6 c h^3. Here c = q = 0.1 and h = 0.05, so the answer is
6 * 0.1 * 0.05^3 = 7.5e-5, whatever the station knew. If the coasting step
left out the random pushes, this number would be zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim.coop.simulator import (
    CoopLinkConfig,
    CoopSwarmConfig,
    CoopSwarmSimulator,
)

# ---------------------------------------------------------------------------
# Plain-numpy helpers, written from the textbook equations. None of them
# reads anything from the simulator.
# ---------------------------------------------------------------------------

Q = 0.1  # random-push strength, m^2/s^3
R = 0.05**2  # assumed measurement variance per axis, m^2
SPEED = 0.8  # target speed along x, m/s
P0 = np.array([[1.0, 0.0], [0.0, 1.0]])  # starting uncertainty, per axis


def _f(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def _qd(dt):
    return Q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])


def predict(mean, cov, dt):
    """Coast a (position, speed) estimate forward by dt seconds."""
    f = _f(dt)
    return f @ mean, f @ cov @ f.T + _qd(dt)


def update(mean, cov, z):
    """Fold in one position measurement z with the optimal gain."""
    h = np.array([[1.0, 0.0]])
    s = (h @ cov @ h.T)[0, 0] + R
    gain = (cov @ h.T)[:, 0] / s
    mean = mean + gain * (z - mean[0])
    cov = (np.eye(2) - np.outer(gain, h[0])) @ cov
    return mean, cov


def steady_state_posterior(period=0.1, iterations=2000):
    """Position/speed covariance right after an update, once it stops changing."""
    mean, cov = np.zeros(2), P0.copy()
    mean, cov = update(mean, cov, 0.0)
    for _ in range(iterations):
        mean, cov = predict(mean, cov, period)
        mean, cov = update(mean, cov, 0.0)
    return cov


def position_variance_after(age, post):
    """P_pp + 2 P_pv a + P_vv a^2 + q a^3 / 3, per axis."""
    return post[0, 0] + 2.0 * post[0, 1] * age + post[1, 1] * age**2 + Q * age**3 / 3.0


def run_a_age(t, delay=0.17, share_period=0.5, capture_period=0.1):
    """Age of the information (t minus the newest fused capture time)."""
    s = share_period * math.floor((t - delay) / share_period + 1e-9)
    u = capture_period * math.floor((s - delay) / capture_period + 1e-9)
    return t - u


def reference_from_events(raw_events, track_events):
    """Return a function giving the expected (x estimate, per-axis variance) at time t.

    raw_events: (capture_time, receive_time or None); track_events:
    (send_time, receive_time or None). Uses only what actually arrived: the
    newest summary received by t was sent at s, and it holds exactly the
    captures that had reached the station by s. Filter states after each
    capture are stored once so every sample just coasts from the right one.
    """
    arrived = sorted((c, rcv) for c, rcv in raw_events if rcv is not None)
    mean, cov = np.zeros(2), P0.copy()
    clock = 0.0
    states = [(mean, cov, clock)]
    for c, _ in arrived:
        mean, cov = predict(mean, cov, c - clock)
        mean, cov = update(mean, cov, SPEED * c)
        clock = c
        states.append((mean, cov, clock))

    def at(t):
        got = [e for e in track_events if e[1] is not None and e[1] <= t + 1e-9]
        sent_at = max(e[0] for e in got)
        count = sum(1 for _, rcv in arrived if rcv <= sent_at + 1e-9)
        # Fixed delay keeps arrivals in capture order, so the arrived ones are a prefix.
        assert all(rcv <= sent_at + 1e-9 for _, rcv in arrived[:count])
        m, c_, k = states[count]
        m, c_ = predict(m, c_, t - k)
        return m[0], c_[0, 0]

    return at


# ---------------------------------------------------------------------------
# Scenario setup and output reading (the only places the simulator is used).
# ---------------------------------------------------------------------------

COMMON = {
    "n_agents": 1,
    "seed": 23,
    "control_dt_s": 0.05,
    "sensor_rate_hz": 10.0,
    "noise_std_m": 0.0,
    "estimator": "cv",
    "fusion": "central",
    "decision": "onboard",
    "target": "cv",
    "target_params": (("speed", 0.8),),
    "dropouts": (),
    "heartbeat_timeout_s": 5.0,
}


def _run_b_config(duration_s):
    return CoopSwarmConfig(
        **COMMON,
        duration_s=duration_s,
        log_every=2,
        link=CoopLinkConfig(
            delay_s=0.17,
            loss_probability=0.3,
            loss_model="independent",
            share_rate_hz=10.0,
        ),
    )


def _samples(run):
    """(time, reported age, reported trace, reported error) for t >= 1 s."""
    rows = []
    for step in run["steps"]:
        t = float(step["time_s"])
        if t >= 1.0 - 1e-9:
            row = step["agents"][0]
            rows.append((t, row["target_age_s"], row["fused_cov_trace_m2"], row["truth_error_m"]))
    return rows


def _events(run):
    raw = [(e["send_time"], e["receive_time"]) for e in run["link_events"] if e["kind"] == "raw"]
    trk = [(e["send_time"], e["receive_time"]) for e in run["link_events"] if e["kind"] == "track"]
    return raw, trk


@pytest.fixture(scope="module")
def run_a():
    cfg = CoopSwarmConfig(
        **COMMON,
        duration_s=8.0,
        log_every=1,
        link=CoopLinkConfig(
            delay_s=0.17,
            loss_probability=0.0,
            loss_model="none",
            share_rate_hz=2.0,
        ),
    )
    return CoopSwarmSimulator(cfg).run().as_dict()


@pytest.fixture(scope="module")
def run_b():
    return CoopSwarmSimulator(_run_b_config(50.0)).run().as_dict()


@pytest.fixture(scope="module")
def rows_a(run_a):
    return _samples(run_a)


@pytest.fixture(scope="module")
def rows_b(run_b):
    return _samples(run_b)


# ---------------------------------------------------------------------------
# Run A: pure latency
# ---------------------------------------------------------------------------


def test_run_a_age_follows_the_two_clock_round_trip(rows_a):
    """Reported age equals delay in + clock wait + delay out + hold time, in 0.40 to 0.85 s steps."""
    assert len(rows_a) == 141
    for t, age, _, _ in rows_a:
        assert age == pytest.approx(run_a_age(t), abs=1e-9)
    values = {round(age, 6) for _, age, _, _ in rows_a}
    assert values == {round(0.40 + 0.05 * i, 6) for i in range(10)}


def test_steady_state_constants_match_the_derivation():
    """Settled covariance after each update has the hand-derived entries."""
    post = steady_state_posterior()
    assert post[0, 0] == pytest.approx(1.1718e-3, abs=5e-8)
    assert post[0, 1] == pytest.approx(3.6445e-3, abs=5e-8)
    assert post[1, 1] == pytest.approx(2.7152e-2, abs=5e-7)
    assert post[0, 1] == pytest.approx(post[1, 0], rel=1e-12)
    # More iterations change nothing, so it really has settled.
    assert steady_state_posterior(iterations=4000) == pytest.approx(post, rel=1e-12)


def test_run_a_trace_grows_with_age_like_the_cubic(rows_a):
    """Three axes of (P_pp + 2 P_pv a + P_vv a^2 + q a^3/3) predict the trace once the filter has settled."""
    post = steady_state_posterior()
    late = [r for r in rows_a if r[0] >= 4.0 - 1e-9]
    assert len(late) == 81
    for t, _, trace, _ in late:
        expected = 3.0 * position_variance_after(run_a_age(t), post)
        assert trace == pytest.approx(expected, rel=1e-6)


def test_run_a_third_difference_is_six_q_h_cubed(rows_a):
    """Four steps on one packet: constants, line and parabola cancel, leaving 6 q h^3 = 7.5e-5."""
    h = 0.05
    expected = 6.0 * Q * h**3
    assert expected == pytest.approx(7.5e-5, rel=1e-12)
    windows = 0
    for i in range(len(rows_a) - 3):
        ages = [rows_a[i + j][1] for j in range(4)]
        same_packet = all(abs(ages[j + 1] - ages[j] - h) < 1e-9 for j in range(3))
        if not same_packet:
            continue
        tr = [rows_a[i + j][2] for j in range(4)]
        third = tr[3] - 3.0 * tr[2] + 3.0 * tr[1] - tr[0]
        assert third == pytest.approx(expected, abs=1e-9)
        windows += 1
    # 14 holds of 10 steps, 7 windows each, minus holds that start before 1 s.
    assert windows >= 90


def test_run_a_straight_line_is_followed_almost_exactly(rows_a):
    """A straight path with no noise leaves only the start-up error, which has died out by 4 s."""
    for t, _, _, err in rows_a:
        if t >= 4.0 - 1e-9:
            assert err < 1e-5


# ---------------------------------------------------------------------------
# Run B: latency plus loss
# ---------------------------------------------------------------------------


def test_run_b_share_of_best_case_ages_matches_both_links_surviving(rows_b):
    """The minimum age 0.4 s needs one raw and one summary packet to arrive: chance 0.7^2 = 0.49."""
    assert len(rows_b) == 491
    best = sum(abs(age - 0.4) < 1e-9 for _, age, _, _ in rows_b) / len(rows_b)
    assert best == pytest.approx(0.49, abs=0.090)


def test_run_b_mean_age_matches_two_geometric_waits(rows_b):
    """Mean age is 0.4 s plus 0.1 s times two geometric counts: 0.4 + 0.2 * 0.3 / 0.7 = 0.48571 s."""
    ages = np.array([age for _, age, _, _ in rows_b])
    assert ages.mean() == pytest.approx(0.4 + 0.1 * 2 * 0.3 / 0.7, abs=0.029)
    # Ages never go below the no-loss floor and move in whole 0.1 s steps.
    assert ages.min() >= 0.4 - 1e-9
    steps = (ages - 0.4) / 0.1
    assert np.abs(steps - np.round(steps)).max() < 1e-9


# ---------------------------------------------------------------------------
# Exact per-sample reference, driven by which packets actually arrived
# ---------------------------------------------------------------------------


def _check_against_reference(run, rows):
    raw, trk = _events(run)
    expected_at = reference_from_events(raw, trk)
    for t, _, trace, err in rows:
        x_ref, var_ref = expected_at(t)
        assert trace == pytest.approx(3.0 * var_ref, rel=1e-9), t
        assert err == pytest.approx(abs(x_ref - SPEED * t), abs=1e-9), t


def test_run_a_every_sample_matches_the_event_log_reference(run_a, rows_a):
    """Replaying the arrived packets through a plain 2x2 filter reproduces each reported trace and error."""
    _check_against_reference(run_a, rows_a)


def test_run_b_every_sample_matches_the_event_log_reference(run_b, rows_b):
    """Same replay with random loss: each sample's trace and error follow from which packets got through."""
    raw, trk = _events(run_b)
    assert any(r is None for _, r in raw) and any(r is None for _, r in trk)
    _check_against_reference(run_b, rows_b)


# ---------------------------------------------------------------------------
# A real bug the checks above had to stop short of
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason=(
        "The drone's capture clock is built by adding 0.1 s over and over, so rounding "
        "error piles up; near t = 54.2 s a capture is stamped a hair later than the "
        "state packet sent in the same step, and the link refuses packets whose send "
        "times go backwards (sim/link.py:166). A 60 s run should finish."
    ),
)
def test_sixty_second_loss_run_finishes_without_link_time_error():
    """A 60 s run with the Run B settings should complete, not crash at about 54.2 s."""
    run = CoopSwarmSimulator(_run_b_config(60.0)).run().as_dict()
    assert run["steps"][-1]["time_s"] >= 59.0
