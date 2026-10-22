import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sim.trace import EspNowTrace
from tools.import_espnow_log import (
    EchoLogError,
    parse_echoer_log,
    parse_sender_log,
    read_echo_logs,
)

SENDER_META = [
    "# zephyr-espnow-echo-log-1", "# role=sender", "# board=A", "# peer=B",
    "# session=a-boot-0001", "# firmware=echo-bench-0.1", "# payload_bytes=32",
    "# channel=1", "# rate_hz=50", "seq,tx_us,echo_us",
]
ECHOER_META = [
    "# zephyr-espnow-echo-log-1", "# role=echoer", "# board=B", "# peer=A",
    "# session=b-boot-0001", "seq,rx_us,echo_tx_us,rssi_dbm",
]


def sender_text(rows=range(8), ending="\n"):
    lines = SENDER_META[:]
    for seq in rows:
        echo = {2: "1003000", 6: "1007000"}.get(seq, "")
        lines.append(f"{seq},{1000000 + seq * 1000},{echo}")
    return ending.join(lines) + ending


def echoer_text(ending="\n"):
    lines = ECHOER_META + [
        "2,2002000,2002100,-42",   # first arrival, echoed
        "1,2003000,, -43".replace(" ", ""),  # out of order, return echo lost
        "1,2004000,, -44".replace(" ", ""),  # duplicate arrival
        "4,2005000,,-45",         # sender saw no echo, but B saw packet
        "6,2006000,2006100,-46",   # echoed
    ]
    return ending.join(lines) + ending


def test_pair_labels_every_situation_and_preserves_arrival_details(tmp_path):
    trace = read_echo_logs(_write(tmp_path, "a.log", sender_text()), _write(tmp_path, "b.log", echoer_text()), trace_id="pair", status="synthetic")
    packets = {p.packet_id: p for p in trace.packets}
    # seq 0 unknown before range, 1 received with lost echo, 2 received with echo,
    # 3 lost inside range, 4 received with lost echo, 5 lost inside range,
    # 6 received with echo and out-of-order context absent, 7 unknown after range.
    logical = [p for p in trace.packets if not p.duplicate]
    assert [p.outcome for p in logical] == ["unknown", "received", "received", "lost", "received", "lost", "received", "unknown"]
    assert packets["a-boot-0001:0"].unknown_reason == "before_echoer_range"
    assert packets["a-boot-0001:7"].unknown_reason == "after_echoer_range"
    assert packets["a-boot-0001:3"].loss_reason == "not_in_echoer_log"
    assert packets["a-boot-0001:2"].echo_time_s == pytest.approx(1.003)
    assert packets["a-boot-0001:2"].echo_turnaround_s == pytest.approx(0.0001)
    assert packets["a-boot-0001:2"].receive_time_s == pytest.approx(2.002)
    assert packets["a-boot-0001:2"].rssi_dbm == -42
    assert packets["a-boot-0001:1"].rssi_dbm == -43
    assert packets["a-boot-0001:1"].out_of_order is True
    duplicate = packets["a-boot-0001:1:dup1"]
    assert duplicate.duplicate is True and duplicate.outcome == "received"
    assert duplicate.receive_time_s == pytest.approx(2.004)
    assert duplicate.echo_time_s is None and duplicate.echo_turnaround_s is None
    assert duplicate.send_time_s == pytest.approx(1.001)


def _write(base, name, text):
    path = base / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


@pytest.fixture
def tmp_logs(tmp_path):
    def make(name, text):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8", newline="")
        return path
    return make


@pytest.mark.parametrize("bad,func,label,line", [
    ("", parse_sender_log, "sender", 1),
    ("\ufeff" + sender_text(), parse_sender_log, "sender", 1),
    (sender_text().replace("\n", "\n\n", 1), parse_sender_log, "sender", 2),
    (sender_text().replace("# role=sender", "# role=echoer"), parse_sender_log, "sender", 2),
    (sender_text().replace("# channel=1", "# channel=15"), parse_sender_log, "sender", 8),
    (sender_text().replace("# payload_bytes=32", "# payload_bytes=-1"), parse_sender_log, "sender", 7),
    (sender_text().replace("# rate_hz=50", "# rate_hz=0"), parse_sender_log, "sender", 9),
    (sender_text().replace("seq,tx_us,echo_us", "seq,tx_us,bad"), parse_sender_log, "sender", 10),
    (sender_text().replace("0,1000000,", "+0,1000000,"), parse_sender_log, "sender", 11),
    (sender_text().replace("1,1001000,", "1,999000,"), parse_sender_log, "sender", 12),
    (sender_text().replace("0,1000000,", "0,1000000,999999"), parse_sender_log, "sender", 11),
    (sender_text().replace("0,1000000,", "0,1000000"), parse_sender_log, "sender", 11),
    (sender_text().replace("# firmware=echo-bench-0.1", "# firmware=echo-bench-0.1\n# firmware=again"), parse_sender_log, "sender", 7),
    (sender_text().replace("# rate_hz=50", "# nope=x"), parse_sender_log, "sender", 9),
    (sender_text().replace("# role=sender", "# board=A\n# role=sender"), parse_sender_log, "sender", 4),
    (sender_text().replace("0,1000000,", "0,1000000,\t"), parse_sender_log, "sender", 11),
    (sender_text().replace("\n2,1002000,1003000", "\n# comment\n2,1002000,1003000"), parse_sender_log, "sender", 13),
    (sender_text(rows=range(2)).replace("1,1001000,", "3,1001000,"), parse_sender_log, "sender", 12),
])
def test_rejections(bad, func, label, line):
    with pytest.raises(EchoLogError) as exc:
        func(bad, label)
    assert f"{label}:{line}:" in str(exc.value)


def test_header_only_logs_are_rejected_and_sender_only_is_unknown(tmp_path):
    with pytest.raises(EchoLogError, match="echoer:7: no packets"):
        parse_echoer_log("\n".join(ECHOER_META) + "\n")
    with pytest.raises(EchoLogError, match="sender:11: no packets"):
        parse_sender_log("\n".join(SENDER_META) + "\n")
    trace = read_echo_logs(_write(tmp_path, "sender.log", sender_text(rows=range(8)).replace("2,1002000,1003000", "2,1002000,").replace("6,1006000,1007000", "6,1006000,")), None, trace_id="solo", status="synthetic")
    assert all(p.outcome == "unknown" for p in trace.packets)


def test_crlf_is_accepted():
    parsed = parse_sender_log(sender_text(ending="\r\n"))
    assert len(parsed.rows) == 8


def test_cli_synthetic_and_measured_requirements(tmp_path):
    sender = tmp_path / "sender.log"; sender.write_text(sender_text().replace("2,1002000,1003000", "2,1002000,").replace("6,1006000,1007000", "6,1006000,"), encoding="utf-8", newline="")
    out = tmp_path / "trace.json"
    script = Path(__file__).parents[2] / "tools" / "import_espnow_log.py"
    py = sys.executable
    result = subprocess.run([py, str(script), "--sender", str(sender), "--trace-id", "cli", "--status", "synthetic", "--out", str(out)], check=False, capture_output=True, text=True, cwd=Path(__file__).parents[2], env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])})
    assert result.returncode == 0, result.stderr
    decoded = json.loads(out.read_text(encoding="utf-8"))
    EspNowTrace.from_path(out)
    schema = json.loads((Path(__file__).parents[2] / "sim" / "trace_schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(decoded)
    missing = subprocess.run([py, str(script), "--sender", str(sender), "--trace-id", "cli", "--status", "measured", "--out", str(out), "--source", "bench"], check=False, capture_output=True, text=True, cwd=Path(__file__).parents[2], env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])})
    assert missing.returncode == 2 and "operator" in missing.stderr
    measured = subprocess.run([py, str(script), "--sender", str(sender), "--trace-id", "cli", "--status", "measured", "--out", str(out), "--source", "bench", "--operator", "A", "--captured-at", "2026-10-20T12:00:00-06:00", "--code-revision", "abc", "--instrument-model", "m"], check=False, capture_output=True, text=True, cwd=Path(__file__).parents[2], env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])})
    assert measured.returncode == 0, measured.stderr
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert prov["raw_sha256"] == hashlib.sha256(sender.read_bytes()).hexdigest()


# Rules added after the first code review. Line numbers are worked out by hand:
# SENDER_META is 10 lines, so data row k (0-based) is line 11 + k; ECHOER_META
# is 6 lines, so echoer data row k is line 7 + k.
@pytest.mark.parametrize("bad,func,label,line", [
    (sender_text()[:-1], parse_sender_log, "sender", 18),  # cut off: no final newline
    (sender_text().replace("2,1002000,", "2,1002000\x0b,", 1), parse_sender_log, "sender", 13),
    (sender_text().replace("0,1000000,", "0,10000000000000000000,"), parse_sender_log, "sender", 11),
    (sender_text().replace("0,1000000,", "0," + "9" * 400 + ","), parse_sender_log, "sender", 11),
    (sender_text().replace("# peer=B", "# peer=A"), parse_sender_log, "sender", 4),
    (sender_text().replace("# rate_hz=50", "# rate_hz=1_0"), parse_sender_log, "sender", 9),
    (sender_text().replace("# firmware=echo-bench-0.1", "# firmware=bench\x00"), parse_sender_log, "sender", 6),
    (echoer_text().replace("2,2002000,2002100,-42", "2,2002000,2002100,-0"), parse_echoer_log, "echoer", 7),
    (echoer_text().replace("1,2004000,,-44", "1,2004000,2004010,-44"), parse_echoer_log, "echoer", 9),
])
def test_rejections_added_after_review(bad, func, label, line):
    with pytest.raises(EchoLogError) as exc:
        func(bad, label)
    assert f"{label}:{line}:" in str(exc.value)


def test_cross_file_errors_name_the_file_and_line(tmp_path):
    sender = _write(tmp_path, "a.log", sender_text())
    # Echoer board does not match the sender's peer: points at the board line (3).
    with pytest.raises(EchoLogError, match="b.log:3: board=C"):
        read_echo_logs(sender, _write(tmp_path, "b.log", echoer_text().replace("# board=B", "# board=C")), trace_id="x", status="synthetic")
    # seq 2 round trip on A is 1003000 - 1002000 = 1000 us; B's hold of 2000 us is impossible.
    # seq 2 is sender data row 2, line 13.
    with pytest.raises(EchoLogError, match="a.log:13: echoer held seq 2 for 2000 us"):
        read_echo_logs(sender, _write(tmp_path, "b.log", echoer_text().replace("2,2002000,2002100", "2,2002000,2004000")), trace_id="x", status="synthetic")
    # A logged an echo for seq 6 (line 17) but B shows no echo for it.
    with pytest.raises(EchoLogError, match="a.log:17: echo logged for seq 6"):
        read_echo_logs(sender, _write(tmp_path, "b.log", echoer_text().replace("6,2006000,2006100", "6,2006000,")), trace_id="x", status="synthetic")
    # B logged seq 9, which A never sent: echoer row 0 is line 7.
    with pytest.raises(EchoLogError, match="b.log:7: seq 9 does not appear"):
        read_echo_logs(sender, _write(tmp_path, "b.log", echoer_text().replace("2,2002000,2002100,-42", "9,2002000,,-42")), trace_id="x", status="synthetic")


def test_cli_reports_bad_input_without_a_traceback(tmp_path, capsys):
    from tools.import_espnow_log import main

    sender = _write(tmp_path, "a.log", sender_text().replace("0,1000000,", "0," + "9" * 400 + ","))
    assert main(["--sender", str(sender), "--trace-id", "x", "--status", "synthetic", "--out", str(tmp_path / "t.json")]) == 2
    assert "a.log:11:" in capsys.readouterr().err
