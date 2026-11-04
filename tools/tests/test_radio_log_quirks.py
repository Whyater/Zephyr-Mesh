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
