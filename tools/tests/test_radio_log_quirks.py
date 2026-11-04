"""Raw-log quirks a real ESP-NOW echo bench can produce.

Each case is a thing a real serial capture can do that the synthetic fixture
never does. For each one the chain (extract, import, fit) must either handle
it correctly or stop with a message that says what went wrong and what to do
at the bench. Nothing here touches a device.

Hand numbers used below:
* A 32-bit microsecond counter wraps at 2**32 us = 4294.967296 s = 71.58 min.
* esp_timer_get_time() printed with a signed 32-bit format (%d / %ld) goes
  negative after 2**31 us = 2147.483648 s = 35.79 min.
"""
from __future__ import annotations

import pytest

from tools import extract_echo_log as ex
from tools.fit_espnow_log import fit_echo_trace
from tools.import_espnow_log import (
    EchoLogError,
    logs_to_trace_dict,
    parse_echoer_log,
    parse_sender_log,
)
from sim.trace import EspNowTrace

WRAP_US = 2 ** 32

SENDER_META = [
    "# zephyr-espnow-echo-log-1", "# role=sender", "# board=A", "# peer=B",
    "# session=a-boot-0001", "# firmware=echo-bench-0.1", "seq,tx_us,echo_us",
]
ECHOER_META = [
    "# zephyr-espnow-echo-log-1", "# role=echoer", "# board=B", "# peer=A",
    "# session=b-boot-0001", "seq,rx_us,echo_tx_us,rssi_dbm",
]
# Data row k (0-based) is on line len(META) + 1 + k.
S0 = len(SENDER_META) + 1
E0 = len(ECHOER_META) + 1


def _text(meta, rows):
    return "\n".join(meta + rows) + "\n"


def bench(n=40, *, tx0=1_000_000, rx0=500_000, period=20_000, out_us=2_000, back_us=2_000, hold_us=40,
          lost_out=(), lost_back=(), seq0=0):
    """Sender and echoer row lists for a clean run (all times in whole us)."""
    s_rows, e_rows = [], []
    for k in range(n):
        seq = seq0 + k
        tx = tx0 + k * period
        if k in lost_out:
            s_rows.append(f"{seq},{tx},")
            continue
        rx = rx0 + k * period + out_us
        etx = rx + hold_us
        e_rows.append(f"{seq},{rx},{etx},-45")
        echo = "" if k in lost_back else str(tx + out_us + hold_us + back_us)
        s_rows.append(f"{seq},{tx},{echo}")
    return s_rows, e_rows


def _trace(s_rows, e_rows):
    sender = parse_sender_log(_text(SENDER_META, s_rows), "sender.csv")
    echoer = parse_echoer_log(_text(ECHOER_META, e_rows), "echoer.csv")
    return EspNowTrace.from_dict(logs_to_trace_dict(sender, echoer, trace_id="quirk", status="synthetic"))


# ---------------------------------------------------------------- clock wrap

def test_hand_numbers_for_the_wrap():
    assert WRAP_US / 1e6 == pytest.approx(4294.967296)
    assert WRAP_US / 1e6 / 60 == pytest.approx(71.58, abs=0.005)
    assert 2 ** 31 / 1e6 / 60 == pytest.approx(35.79, abs=0.005)


def test_sender_tx_us_wrapping_at_2_32_is_named_as_a_32_bit_clock():
    # Rows straddle the wrap: the last stamp before it is 2**32 - 15000 us.
    s_rows = [f"{k},{(WRAP_US - 35_000 + 20_000 * k) % WRAP_US}," for k in range(4)]
    assert s_rows[2] == "2,5000,"  # the counter wrapped between rows 1 and 2
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, s_rows), "sender.csv")
    message = str(exc.value)
    assert message.startswith(f"sender.csv:{S0 + 2}: tx_us must always increase")
    assert "32-bit" in message and "71.58 min" in message and "esp_timer_get_time" in message


def test_echo_us_wrapping_inside_one_round_trip_is_named_as_a_32_bit_clock():
    tx = WRAP_US - 1_000
    echo = (tx + 4_040) - WRAP_US  # the echo came back after the counter wrapped
    rows = [f"0,{tx},{echo}"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    assert "echo_us cannot precede tx_us" in str(exc.value)
    assert "32-bit" in str(exc.value)


def test_echoer_rx_us_wrapping_is_named_as_a_32_bit_clock():
    rows = [f"0,{WRAP_US - 10_000},{WRAP_US - 9_960},-45", "1,10000,10040,-45"]
    with pytest.raises(EchoLogError) as exc:
        parse_echoer_log(_text(ECHOER_META, rows), "echoer.csv")
    assert str(exc.value).startswith(f"echoer.csv:{E0 + 1}: rx_us must never decrease")
    assert "32-bit" in str(exc.value)


def test_a_negative_stamp_says_the_64_bit_clock_was_printed_with_a_32_bit_format():
    # esp_timer_get_time() is int64; printf("%ld") of it on the ESP32 keeps 32 bits, signed.
    rows = ["0,2147473648,", "1,-2147473648,"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    assert str(exc.value).startswith(f"sender.csv:{S0 + 1}:")
    assert "negative" in str(exc.value) and "%lld" in str(exc.value)


def test_an_ordinary_backward_step_does_not_blame_the_clock_width():
    rows = ["0,1000000,", "1,999000,"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    assert "32-bit" not in str(exc.value)


# ---------------------------------------------------------------- sequence quirks

def test_sender_seq_wrapping_at_16_bits_is_named():
    rows = ["65534,1000000,", "65535,1020000,", "0,1040000,"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    assert str(exc.value).startswith(f"sender.csv:{S0 + 2}: seq jumps from 65535 to 0")
    assert "16-bit" in str(exc.value)


def test_sender_seq_going_back_to_zero_mid_block_points_at_a_restart():
    rows = ["0,1000000,", "1,1020000,", "2,1040000,", "0,1060000,"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    assert "back to 0" in str(exc.value) and "--split" in str(exc.value)


def test_a_repeated_sender_row_is_called_a_repeat_not_lost_lines():
    rows = ["0,1000000,", "1,1020000,", "1,1020000,", "2,1040000,"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    message = str(exc.value)
    assert message.startswith(f"sender.csv:{S0 + 2}:")
    assert "repeated" in message and "lost serial lines" not in message


def test_sender_rows_printed_out_of_seq_order_say_so():
    # A firmware that prints a timed-out packet late puts seq 1 after seq 2 and 3.
    rows = ["0,1000000,1004040", "2,1040000,1044040", "3,1060000,1064040", "1,1020000,"]
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, rows), "sender.csv")
    message = str(exc.value)
    assert message.startswith(f"sender.csv:{S0 + 1}:")
    assert "seq order" in message and "seq 1" in message


def test_sender_reboot_with_a_deduplicating_echoer_is_not_silently_merged():
    """Board A reboots; board B keeps running and (as the format says) does not echo a seq twice.

    Session 1: A sends seq 0..19, B echoes all. A reboots (its clock restarts) and sends seq
    0..39. B sees 0..19 again, logs them as repeats with no echo, and echoes 20..39. Importing
    A's second block with B's whole log used to give a clean-looking trace: seq 0..19
    "received" with B's session-1 arrival times, a 50% return loss, and 20 fake duplicates.
    """
    _, e1 = bench(20, rx0=500_000)
    gap_us = 20 * 20_000 + 800_000  # B's clock runs on through A's reboot (about 0.8 s)
    s2, e2 = bench(40, tx0=300_000, rx0=500_000 + gap_us)
    e2 = [row.split(",") for row in e2]
    e2 = [f"{seq},{rx},{'' if int(seq) < 20 else etx},{rssi}" for seq, rx, etx, rssi in e2]
    s2 = [row if int(row.split(",")[0]) >= 20 else ",".join(row.split(",")[:2]) + "," for row in s2]
    sender = parse_sender_log(_text(SENDER_META, s2), "sender.csv")
    echoer = parse_echoer_log(_text(ECHOER_META, e1 + e2), "echoer.csv")
    with pytest.raises(EchoLogError) as exc:
        logs_to_trace_dict(sender, echoer, trace_id="reboot", status="synthetic")
    message = str(exc.value)
    assert message.startswith(f"echoer.csv:{E0 + 20}: seq 0 arrives again")
    assert "reboot" in message


def test_sender_first_block_with_an_echoer_log_spanning_a_reboot_says_why():
    s1, e1 = bench(20)
    _, e2 = bench(40, rx0=500_000 + 1_200_000)
    sender = parse_sender_log(_text(SENDER_META, s1), "sender.csv")
    echoer = parse_echoer_log(_text(ECHOER_META, e1 + [r.split(",")[0] + "," + r.split(",")[1] + ",,-45"
                                                         for r in e2]), "echoer.csv")
    with pytest.raises(EchoLogError) as exc:
        logs_to_trace_dict(sender, echoer, trace_id="reboot", status="synthetic")
    assert "reboot" in str(exc.value)


def test_a_real_radio_duplicate_milliseconds_later_is_still_accepted():
    s_rows, e_rows = bench(20, lost_out={7})
    seq, rx, _, rssi = e_rows[4].split(",")
    e_rows.insert(5, f"{seq},{int(rx) + 1_000},,{rssi}")  # MAC retry duplicate, 1 ms later
    trace = _trace(s_rows, e_rows)
    dups = [p for p in trace.packets if p.duplicate]
    assert len(dups) == 1 and dups[0].seq == 4
    fit = fit_echo_trace(trace, n_boot=50)
    assert fit["loss"]["n_duplicates_excluded"] == 1
    assert fit["loss"]["n"] == 20 and fit["loss"]["n_lost"] == 1


def test_out_of_order_arrivals_keep_their_own_times():
    s_rows, e_rows = bench(10)
    e_rows[3], e_rows[4] = e_rows[4], e_rows[3]  # seq 4 overtakes seq 3 at B
    seq4, rx4, etx4, rssi = e_rows[3].split(",")
    seq3, rx3, etx3, _ = e_rows[4].split(",")
    e_rows[3] = f"{seq4},{int(rx3)},{int(rx3) + 40},{rssi}"  # keep rx_us nondecreasing
    e_rows[4] = f"{seq3},{int(rx3) + 500},{int(rx3) + 540},{rssi}"
    trace = _trace(s_rows, e_rows)
    by_seq = {p.seq: p for p in trace.packets}
    assert by_seq[3].out_of_order is True and by_seq[4].out_of_order is None
    assert by_seq[3].receive_time_s == pytest.approx((int(rx3) + 500) / 1e6)
    assert all(p.outcome == "received" for p in trace.packets)


# ---------------------------------------------------------------- line endings and cut lines

def test_crlf_logs_parse_the_same_as_lf_logs():
    s_rows, e_rows = bench(10, lost_out={3})
    lf = _trace(s_rows, e_rows)
    sender = parse_sender_log(_text(SENDER_META, s_rows).replace("\n", "\r\n"), "sender.csv")
    echoer = parse_echoer_log(_text(ECHOER_META, e_rows).replace("\n", "\r\n"), "echoer.csv")
    crlf = EspNowTrace.from_dict(logs_to_trace_dict(sender, echoer, trace_id="quirk", status="synthetic"))
    assert crlf.as_dict()["packets"] == lf.as_dict()["packets"]


def test_cr_only_line_endings_are_named():
    s_rows, _ = bench(3)
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, s_rows).replace("\n", "\r"), "sender.csv")
    assert "\\r" in str(exc.value) and "line endings" in str(exc.value)


def test_a_cut_final_line_is_rejected_by_the_importer_and_dropped_by_the_extractor(tmp_path):
    s_rows, _ = bench(5)
    text = _text(SENDER_META, s_rows)
    cut = text + "5,1100000,11"  # the capture stopped mid-row
    with pytest.raises(EchoLogError, match="no newline"):
        parse_sender_log(cut, "sender.csv")
    raw = tmp_path / "raw.txt"
    raw.write_bytes(cut.replace("\n", "\r\n").encode())
    audit = ex.extract_file(raw, out=tmp_path / "clean.csv")
    assert (tmp_path / "clean.csv").read_text(encoding="utf-8") == text
    assert audit["truncated_final_line"]["text"] == "5,1100000,11"


# ---------------------------------------------------------------- noise and partial captures

@pytest.mark.parametrize("junk", [
    "W (1234) wifi:wifi send failed",
    "E (88) ESPNOW: send queue full",
    "ets Jul 29 2019 12:21:46",
])
def test_serial_noise_inside_a_block_is_named_as_not_a_data_row(tmp_path, junk):
    s_rows, _ = bench(6)
    lines = SENDER_META + s_rows[:3] + [junk] + s_rows[3:]
    raw = tmp_path / "raw.txt"
    raw.write_bytes(("boot banner\r\n" + "\r\n".join(lines) + "\r\n").encode())
    with pytest.raises(ex.ExtractError) as exc:
        ex.extract_file(raw, out=tmp_path / "clean.csv")
    message = str(exc.value)
    assert message.startswith(f"raw.txt:{1 + len(SENDER_META) + 4}:")
    assert "not a data row" in message and "esp_log_level_set" in message


def test_capture_that_starts_mid_line_before_the_header_is_fine(tmp_path):
    s_rows, _ = bench(6)
    text = _text(SENDER_META, s_rows)
    raw = tmp_path / "raw.txt"
    raw.write_bytes(b"0x40080400 load:0x3fff0030,len:7\xff\r\n" + text.replace("\n", "\r\n").encode())
    audit = ex.extract_file(raw, out=tmp_path / "clean.csv")
    assert (tmp_path / "clean.csv").read_text(encoding="utf-8") == text
    assert [d["reason"] for d in audit["dropped"]] == ["before_first_block"]


def test_capture_that_starts_after_the_header_says_to_capture_from_boot(tmp_path):
    s_rows, _ = bench(6)
    raw = tmp_path / "raw.txt"
    raw.write_bytes(("00,1060000,\r\n" + "\r\n".join(s_rows) + "\r\n").encode())  # starts mid-row
    with pytest.raises(ex.ExtractError) as exc:
        ex.extract_file(raw, out=tmp_path / "clean.csv")
    assert "no line is exactly" in str(exc.value)
    assert "reset the board" in str(exc.value)


# ---------------------------------------------------------------- one-way asymmetry

def test_asymmetric_legs_give_the_mean_of_both_legs_and_say_it_is_an_estimate():
    # Real legs: 1000 us out, 5000 us back. An echo can only see the sum.
    s_rows, e_rows = bench(60, out_us=1_000, back_us=5_000, lost_out={10, 30})
    fit = fit_echo_trace(_trace(s_rows, e_rows), n_boot=200)
    one_way = fit["delay"]["one_way_estimate"]
    assert one_way["delay_s"]["value"] == pytest.approx(0.003)   # (1000 + 5000) / 2 us, not 1000 us
    assert fit["delay"]["rtt_s"]["mean"]["value"] == pytest.approx(0.006)
    assert any("same mean time" in a for a in fit["delay"]["assumptions"])
    assert any("(d_back - d_out) / 2" in c for c in fit["delay"]["consequences"])
    assert "one_way_measured" not in str(fit) and "measured one-way" not in str(fit).lower()


# ---------------------------------------------------------------- repair: formatting mistakes and short sessions

@pytest.mark.parametrize("row, field, cause", [
    ("0, 1000000, 1004040", "tx_us", "space"),        # spaces after commas
    ("0,1.000000,1.004040", "tx_us", "decimal point"),  # seconds printed as decimals
    ("0,+1000000,", "tx_us", "+ sign"),
])
def test_a_badly_formatted_data_row_names_the_field_not_serial_noise(row, field, cause):
    s_rows, _ = bench(3)
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, s_rows[:1] + [row.replace("0,", "1,", 1)] + s_rows[2:]), "sender.csv")
    message = str(exc.value)
    assert message.startswith(f"sender.csv:{S0 + 1}: {field} ")
    assert cause in message and "serial output" not in message


def test_a_stray_carriage_return_is_named_not_called_serial_noise():
    s_rows, _ = bench(3)
    s_rows[1] += "\r"  # with CRLF endings this row ends \r\r\n
    with pytest.raises(EchoLogError) as exc:
        parse_sender_log(_text(SENDER_META, s_rows).replace("\n", "\r\n"), "sender.csv")
    message = str(exc.value)
    assert message.startswith(f"sender.csv:{S0 + 1}: echo_us ")
    assert "a stray \\r" in message and "serial output" not in message


def test_an_echoer_row_with_spaces_names_the_field():
    _, e_rows = bench(3)
    e_rows[2] = e_rows[2].replace(",", ", ")
    with pytest.raises(EchoLogError) as exc:
        parse_echoer_log(_text(ECHOER_META, e_rows), "echoer.csv")
    assert str(exc.value).startswith(f"echoer.csv:{E0 + 2}: rx_us ") and "space" in str(exc.value)


def _reboot_logs(first_session, reboot_us):
    """A sends seq 0..first_session-1, reboots, then sends seq 0..39; B runs on and logs both sessions."""
    _, e1 = bench(first_session, rx0=500_000)
    s2, e2 = bench(40, tx0=300_000, rx0=500_000 + first_session * 20_000 + reboot_us)
    e2 = [row.split(",") for row in e2]
    e2 = [f"{seq},{rx},{'' if int(seq) < first_session else etx},{rssi}" for seq, rx, etx, rssi in e2]
    s2 = [row if int(row.split(",")[0]) >= first_session else ",".join(row.split(",")[:2]) + "," for row in s2]
    return parse_sender_log(_text(SENDER_META, s2), "sender.csv"), parse_echoer_log(_text(ECHOER_META, e1 + e2),
                                                                                    "echoer.csv")


def test_a_sender_reboot_after_a_very_short_first_session_is_caught():
    """A browns out after 5 packets (0.1 s) and is back 0.5 s later, so each repeat is only 0.6 s late.

    The 1 s duplicate window cannot see this. The clock offset can: rx_us - tx_us is B's clock minus
    A's clock plus the one-way delay. A's clock restarts at the reboot, so the offset jumps by the
    0.1 s session plus the 0.5 s reboot = 0.6 s between seq 4 (session 1) and seq 5 (session 2).
    Seq 5's round trip says at most 4 ms of that can be delay, so the clocks moved at least 0.596 s.
    """
    sender, echoer = _reboot_logs(5, 500_000)
    with pytest.raises(EchoLogError) as exc:
        logs_to_trace_dict(sender, echoer, trace_id="reboot", status="synthetic")
    message = str(exc.value)
    assert message.startswith(f"echoer.csv:{E0 + 5 + 5}:")  # seq 5, the first session-2 arrival
    assert "at least 0.596 s" in message and "seq 4" in message and "reboot" in message


def test_a_repeat_exactly_one_second_later_is_accepted_and_one_microsecond_more_is_refused():
    for extra, refused in ((1_000_000, False), (1_000_001, True)):
        s_rows, e_rows = bench(10)
        seq, rx, _, rssi = e_rows[3].split(",")
        e_rows.append(f"{seq},{int(rx) + extra},,{rssi}")
        e_rows.sort(key=lambda r: int(r.split(",")[1]))
        if refused:
            with pytest.raises(EchoLogError, match="arrives again 1.000 s"):
                _trace(s_rows, e_rows)
        else:
            assert sum(p.duplicate is True for p in _trace(s_rows, e_rows).packets) == 1


def test_a_late_delivery_and_a_long_loss_stretch_with_drift_still_import():
    """One packet arrives 15 ms late; then 3000 packets (60 s) are lost while B's clock drifts 100 ppm.

    Hand numbers: 100 ppm over 60 s is 6 ms of offset change, and the late packet moves the offset by
    15 ms, both far below a reboot (hundreds of ms).
    """
    s_rows, e_rows = bench(3020, lost_out=set(range(10, 3010)))
    rows = []
    for row in e_rows:
        seq, rx, etx, rssi = (int(v) for v in row.split(","))
        rx += (rx - 500_000) // 10_000  # 100 ppm fast
        etx = rx + 40
        if seq == 5:
            rx, etx = rx + 15_000, etx + 15_000
        rows.append(f"{seq},{rx},{etx},{rssi}")
    s_rows = [r if int(r.split(",")[0]) != 5 else r.rsplit(",", 1)[0] + f",{int(r.rsplit(',', 1)[1]) + 15_000}"
              for r in s_rows]
    trace = _trace(s_rows, rows)
    assert sum(p.outcome == "lost" for p in trace.packets) == 3000


# ---------------------------------------------------------------- tell a reboot from a stall with A's round trip
#
# rx_us - tx_us is (B's clock minus A's clock) + the forward delay. A stall changes the delay; a reboot of A changes
# the clock difference. Board A's own round trip bounds the delay, so only the second can push the clock difference
# outside the window the earlier packets allow.

def _delayed(out_us, *, period=20_000, tx0=1_000_000, rx0=500_000, hold_us=40, back_us=2_000, clock_step=None):
    """Rows for a run where packet k takes out_us[k] to reach B (B prints rows in arrival order).

    ``clock_step=(k0, step_us)`` moves B's clock (rx_us and echo_tx_us) forward by step_us from packet k0 on,
    with no change in any delay: what a clock jump looks like with the delays held fixed.
    """
    s_rows, arrivals = [], []
    for k, out in enumerate(out_us):
        tx = tx0 + k * period
        shift = clock_step[1] if clock_step is not None and k >= clock_step[0] else 0
        rx = rx0 + k * period + out + shift
        arrivals.append((rx, f"{k},{rx},{rx + hold_us},-45"))
        s_rows.append(f"{k},{tx},{tx + out + hold_us + back_us}")
    arrivals.sort(key=lambda a: a[0])
    return s_rows, [row for _, row in arrivals]


def _stall(n, start, peak_us, step_us=20_000, base_us=2_000):
    """Packet ``start`` waits peak_us; the ones queued behind it each wait step_us less (a FIFO draining)."""
    out = [base_us] * n
    for j in range(n - start):
        out[start + j] = max(base_us, peak_us - j * step_us)
    return out


@pytest.mark.parametrize("name, out_us", [
    ("150 ms stall, 7 packets queued behind it", _stall(200, 50, 150_000)),
    ("600 ms stall", _stall(200, 50, 600_000)),
    ("one packet 150 ms late, overtaken by the next 7", [150_000 if k == 60 else 2_000 for k in range(200)]),
    ("queue builds to 296 ms over 1 s, then drains at once",
     [2_000 + (k - 50) * 6_000 if 50 <= k < 100 else 2_000 for k in range(200)]),
])
def test_a_link_stall_is_not_mistaken_for_a_reboot(name, out_us):
    """No reboot here: only the forward delay changes, and A's round trip grows by the same amount."""
    s_rows, e_rows = _delayed(out_us)
    trace = _trace(s_rows, e_rows)
    assert len(trace.packets) == 200 and all(p.outcome == "received" for p in trace.packets)


def test_the_generator_at_30_ms_jitter_imports():
    """The repo's own generator with 30 ms delay spread on both legs (the 100 ms rule refused this, jump 0.161 s)."""
    from tools.make_espnow_echo_fixture import EchoParams, generate_echo_logs
    logs = generate_echo_logs(EchoParams(out_delay_std_s=0.03, back_delay_std_s=0.03), 3)
    trace = logs_to_trace_dict(parse_sender_log(logs.sender_text, "sender.csv"),
                               parse_echoer_log(logs.echoer_text, "echoer.csv"), trace_id="jitter", status="synthetic")
    assert len(trace["packets"]) >= 3000


def test_ten_minutes_with_the_boards_drifting_200_ppm_apart_imports():
    """+100 ppm on A and -100 ppm on B: the clock difference slides 0.12 s over 600 s, smoothly."""
    from tools.make_espnow_echo_fixture import EchoParams, generate_echo_logs
    logs = generate_echo_logs(EchoParams(n_packets=30_000, sender_drift_ppm=100.0, echoer_drift_ppm=-100.0), 7)
    trace = logs_to_trace_dict(parse_sender_log(logs.sender_text, "sender.csv"),
                               parse_echoer_log(logs.echoer_text, "echoer.csv"), trace_id="drift", status="synthetic")
    assert len(trace["packets"]) >= 30_000


@pytest.mark.parametrize("first_session, reboot_us, gap_s", [
    (5, 500_000, "0.596"),   # brownout in the first 0.1 s, back 0.5 s later
    (1, 300_000, "0.316"),   # brownout on the very first send
    (1, 150_000, "0.166"),
    (1, 50_000, "0.066"),    # back in 0.05 s: the old 100 ms rule missed this
])
def test_a_sender_reboot_is_refused_and_named(first_session, reboot_us, gap_s):
    """Hand numbers: B's clock minus A's clock jumps by first_session * 20 ms + reboot_us (A's clock restarts).

    The first session-2 packet's round trip is 4.040 ms and B held it 40 us, so at most 4 ms of the jump can be
    forward delay: the unexplained part is first_session * 20 ms + reboot_us - 4 ms.
    """
    sender, echoer = _reboot_logs(first_session, reboot_us)
    with pytest.raises(EchoLogError) as exc:
        logs_to_trace_dict(sender, echoer, trace_id="reboot", status="synthetic")
    message = str(exc.value)
    assert message.startswith(f"echoer.csv:{E0 + 2 * first_session}:")  # the first session-2 arrival
    assert f"at least {gap_s} s" in message
    assert f"seq {first_session - 1} and seq {first_session}" in message and "rebooted" in message


def test_a_reboot_is_caught_even_when_the_first_echo_after_it_is_lost():
    """Seq 3 is the first packet after the reboot and its echo never got back, so it has no round trip.

    Comparing only neighbours would miss this: seq 4 would be checked against seq 3, which is already in the
    new session. The rule keeps the tightest window from every earlier packet, so seq 4 is still checked
    against seq 2. Unexplained jump: 3 * 20 ms + 0.2 s - 4 ms (seq 4's delay bound) = 0.256 s.
    """
    _, e1 = bench(3, rx0=500_000)
    s2, e2 = bench(40, tx0=300_000, rx0=500_000 + 3 * 20_000 + 200_000, lost_back={3})
    e2 = [row.split(",") for row in e2]
    e2 = [f"{seq},{rx},{'' if int(seq) < 3 else etx},{rssi}" for seq, rx, etx, rssi in e2]
    s2 = [row if int(row.split(",")[0]) >= 3 else ",".join(row.split(",")[:2]) + "," for row in s2]
    sender = parse_sender_log(_text(SENDER_META, s2), "sender.csv")
    echoer = parse_echoer_log(_text(ECHOER_META, e1 + e2), "echoer.csv")
    with pytest.raises(EchoLogError) as exc:
        logs_to_trace_dict(sender, echoer, trace_id="reboot", status="synthetic")
    message = str(exc.value)
    assert message.startswith(f"echoer.csv:{E0 + 3 + 4}:")  # seq 4
    assert "at least 0.256 s" in message and "seq 2 and seq 4" in message


@pytest.mark.parametrize("step_us, refused", [(24_020, False), (24_021, True)])
def test_the_clock_jump_threshold_by_hand(step_us, refused):
    """B's clock jumps by step_us at packet 20 with every delay unchanged.

    Packet 20's round trip (4.040 ms) minus B's hold (40 us) lets 4 ms of the jump be delay. On top of that the
    rule allows 20 ms (stamping slack) plus 1000 ppm of the 20 ms between packets 19 and 20 = 20 us. So
    24.020 ms is the largest jump accepted. This test pins the constants; it restates the rule's arithmetic.
    """
    s_rows, e_rows = _delayed([2_000] * 40, clock_step=(20, step_us))
    if refused:
        with pytest.raises(EchoLogError, match="seq 19 and seq 20"):
            _trace(s_rows, e_rows)
    else:
        assert len(_trace(s_rows, e_rows).packets) == 40


def test_a_clock_gap_that_falls_too_far_says_the_files_do_not_belong_together():
    """Packets 20..29 never reach B (0.2 s gap), then B's clock is 0.1 s behind where it was.

    B's rows still go forward (rx_us of packet 30 is 0.1 s after packet 19's), so only this check sees it. Hand
    numbers: packet 19 pins B's clock minus A's clock to at least -0.502 s; packet 30's offset is -0.598 s, so the
    gap fell by at least 0.096 s, far past the 20 ms allowance.
    """
    s_rows, e_rows = _delayed([2_000] * 40, clock_step=(30, -100_000))
    e_rows = [r for r in e_rows if not 20 <= int(r.split(",")[0]) < 30]
    s_rows = [r if not 20 <= int(r.split(",")[0]) < 30 else ",".join(r.split(",")[:2]) + "," for r in s_rows]
    with pytest.raises(EchoLogError) as exc:
        _trace(s_rows, e_rows)
    message = str(exc.value)
    assert message.startswith(f"echoer.csv:{E0 + 20}:")  # packet 30 is B's 21st row
    assert "fell by at least 0.096 s between seq 19 and seq 30" in message and "same run" in message
