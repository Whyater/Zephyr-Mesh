"""Tests for tools/extract_echo_log.py.

The truth in every test is a log the fixture generator wrote before any
serial noise was added. The noise (boot text, CRLF endings, a cut final line,
a reset) is added here by hand, and the extract must give back the original
bytes, or fail on the line the noise was put on.
"""
import json
from pathlib import Path

import pytest

from tools import extract_echo_log as ex
from tools.import_espnow_log import read_echo_logs
from tools.make_espnow_echo_fixture import EchoParams, generate_echo_logs

FIXTURE = Path(__file__).resolve().parent / "data" / "espnow_echo"

# What an ESP32 prints before the app starts (ROM banner plus bootloader),
# including a blank line and one line of baud-mismatch junk that is not UTF-8.
BANNER = [
    b"\xff\x00\xfe\x8c junk at the wrong baud",
    b"ets Jun  8 2016 00:22:57",
    b"",
    b"rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)",
    b"configsip: 0, SPIWP:0xee",
    b"mode:DIO, clock div:2",
    b"load:0x3fff0030,len:7104",
    b"entry 0x400805cc",
    b"I (29) boot: ESP-IDF v5.2 2nd stage bootloader",
]


def _crlf(lines):
    return b"".join(line + b"\r\n" for line in lines)


def _lines(text):
    return [line.encode("utf-8") for line in text.split("\n")[:-1]]


def _write(path, data):
    path.write_bytes(data)
    return path


def _small_logs(seed, n=40):
    return generate_echo_logs(EchoParams(n_packets=n), seed)


def test_banner_crlf_and_cut_line_give_back_the_fixture_minus_the_cut_row(tmp_path):
    original = (FIXTURE / "sender.csv").read_text(encoding="utf-8")
    lines = _lines(original)
    last = lines[-1]
    cut = last[: len(last) // 2]
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(lines[:-1]) + cut)
    out = tmp_path / "sender.csv"
    audit = ex.extract_file(raw, out=out)

    expected = "\n".join(original.split("\n")[:-2]) + "\n"
    assert out.read_bytes() == expected.encode("utf-8")
    # Every banner line is listed, in order, with its raw line number.
    before = [d for d in audit["dropped"] if d["reason"] == "before_first_block"]
    assert [d["raw_line"] for d in before] == list(range(1, len(BANNER) + 1))
    assert before[1]["text"] == "ets Jun  8 2016 00:22:57"
    # Bytes that are not UTF-8 show as \xNN; NUL is valid UTF-8 so it stays a character.
    assert before[0]["text"].startswith("\\xff\x00\\xfe\\x8c")
    # The cut line is flagged, not kept.
    trunc = audit["truncated_final_line"]
    assert trunc == {"raw_line": len(BANNER) + len(lines), "text": cut.decode("utf-8")}
    assert audit["crlf_lines"] == len(BANNER) + len(lines) - 1
    assert audit["blocks"][0]["data_rows"] == len(lines) - 10 - 1  # 10 header lines in the fixture

    trace = read_echo_logs(out, trace_id="t8", status="synthetic")
    original_rows = len(lines) - 10
    assert len(trace.packets) == original_rows - 1
    assert trace.status == "synthetic"


def test_clean_pair_through_the_extractor_gives_the_same_trace_as_the_fixture(tmp_path):
    clean = {}
    for name in ("sender.csv", "echoer.csv"):
        text = (FIXTURE / name).read_text(encoding="utf-8")
        raw = _write(tmp_path / f"raw-{name}", _crlf(BANNER) + _crlf(_lines(text)))
        (tmp_path / "out").mkdir(exist_ok=True)
        ex.extract_file(raw, out=tmp_path / "out" / name)
        clean[name] = text
        assert (tmp_path / "out" / name).read_text(encoding="utf-8") == text
    got = read_echo_logs(tmp_path / "out" / "sender.csv", tmp_path / "out" / "echoer.csv",
                         trace_id="t8", status="synthetic")
    want = read_echo_logs(FIXTURE / "sender.csv", FIXTURE / "echoer.csv", trace_id="t8", status="synthetic")
    assert got.as_dict() == want.as_dict()


def _raw_with_insert(tmp_path, insert_at, extra):
    """Banner, then the small sender log, with ``extra`` lines put before log line ``insert_at``."""
    lines = _lines(_small_logs(1).sender_text)
    body = lines[:insert_at] + extra + lines[insert_at:]
    return _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(body)), lines


@pytest.mark.parametrize("junk", [
    b"E (1234) wifi: esp_wifi_send failed",
    b"",
    b"12,13",
    b"Guru Meditation Error: Core  1 panic'ed (LoadProhibited)",
])
def test_garbage_inside_the_block_fails_on_its_raw_line(tmp_path, junk):
    raw, _ = _raw_with_insert(tmp_path, 15, [junk])
    out = tmp_path / "out.csv"
    with pytest.raises(ex.ExtractError) as exc:
        ex.extract_file(raw, out=out)
    assert f"raw.txt:{len(BANNER) + 16}:" in str(exc.value)
    assert not out.exists()


def test_invalid_utf8_inside_the_block_fails_on_its_raw_line(tmp_path):
    raw, _ = _raw_with_insert(tmp_path, 12, [b"5,10\xff00,"])
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{len(BANNER) + 13}: .*UTF-8"):
        ex.extract_file(raw, out=tmp_path / "out.csv")


def test_lone_carriage_return_inside_a_row_fails(tmp_path):
    lines = _lines(_small_logs(1).sender_text)
    lines[20] = lines[20].replace(b",", b"\r,", 1)
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(lines))
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{len(BANNER) + 21}:"):
        ex.extract_file(raw, out=tmp_path / "out.csv")


def _reset_capture(tmp_path):
    first, second = _small_logs(1, n=30), _small_logs(2, n=25)
    a, b = _lines(first.sender_text), _lines(second.sender_text)
    data = _crlf(BANNER) + _crlf(a) + _crlf(BANNER[1:]) + _crlf(b)
    return _write(tmp_path / "raw.txt", data), first, second, a


def test_second_format_line_without_split_fails_on_that_line(tmp_path):
    raw, _, _, a = _reset_capture(tmp_path)
    second_format = len(BANNER) + len(a) + len(BANNER) - 1 + 1
    out = tmp_path / "out.csv"
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{second_format}: .*--split"):
        ex.extract_file(raw, out=out)
    assert not out.exists()


def test_split_writes_one_file_per_block_equal_to_each_session(tmp_path):
    raw, first, second, a = _reset_capture(tmp_path)
    out_dir = tmp_path / "blocks"
    audit = ex.extract_file(raw, out_dir=out_dir, split=True)
    assert sorted(p.name for p in out_dir.iterdir()) == ["audit.json", "block-1.csv", "block-2.csv"]
    assert (out_dir / "block-1.csv").read_text(encoding="utf-8") == first.sender_text
    assert (out_dir / "block-2.csv").read_text(encoding="utf-8") == second.sender_text
    reset_text = [d for d in audit["dropped"] if d["reason"] == "reset_boot_text"]
    first_reset_line = len(BANNER) + len(a) + 1
    assert [d["raw_line"] for d in reset_text] == list(range(first_reset_line, first_reset_line + len(BANNER) - 1))
    assert [b["ended_by"] for b in audit["blocks"]] == ["reset", "end_of_capture"]
    assert audit["blocks"][0]["last_row_may_be_cut"] is True
    assert audit["blocks"][1]["last_row_may_be_cut"] is False
    assert json.loads((out_dir / "audit.json").read_text(encoding="utf-8")) == audit
    for name in ("block-1.csv", "block-2.csv"):
        read_echo_logs(out_dir / name, trace_id=name, status="synthetic")


def test_split_still_fails_on_garbage_in_the_middle_of_a_block(tmp_path):
    first, second = _small_logs(1, n=30), _small_logs(2, n=25)
    a, b = _lines(first.sender_text), _lines(second.sender_text)
    a = a[:20] + [b"W (88) wifi: junk"] + a[20:]
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(a) + _crlf(BANNER[1:]) + _crlf(b))
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{len(BANNER) + 21}:"):
        ex.extract_file(raw, out_dir=tmp_path / "blocks", split=True)
    assert not (tmp_path / "blocks").exists()


def test_text_after_the_last_block_fails_even_with_split(tmp_path):
    lines = _lines(_small_logs(1).sender_text)
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(lines) + _crlf([b"I (999) app: test done"]))
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{len(BANNER) + len(lines) + 1}:"):
        ex.extract_file(raw, out_dir=tmp_path / "blocks", split=True)


def test_echoer_block_is_checked_with_the_echoer_rules(tmp_path):
    logs = _small_logs(3)
    lines = _lines(logs.echoer_text)
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(lines))
    audit = ex.extract_file(raw, out=tmp_path / "echoer.csv")
    assert (tmp_path / "echoer.csv").read_text(encoding="utf-8") == logs.echoer_text
    assert audit["blocks"][0]["role"] == "echoer"
    bad = list(lines)
    bad[8] = bad[8] + b",-40"  # a fifth field
    raw = _write(tmp_path / "raw2.txt", _crlf(BANNER) + _crlf(bad))
    with pytest.raises(ex.ExtractError, match=rf"raw2.txt:{len(BANNER) + 9}: expected 4 fields"):
        ex.extract_file(raw, out=tmp_path / "echoer2.csv")


@pytest.mark.parametrize("data,line,message", [
    (b"", 1, "empty"),
    (_crlf(BANNER), 1, "no line is exactly"),
    (_crlf(BANNER + [b"\x00# zephyr-espnow-echo-log-1"]), len(BANNER) + 1, "extra characters"),
])
def test_no_clean_format_line_fails(tmp_path, data, line, message):
    raw = _write(tmp_path / "raw.txt", data)
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{line}: .*{message}"):
        ex.extract_file(raw, out=tmp_path / "out.csv")


def test_same_input_gives_byte_identical_output_and_audit(tmp_path):
    lines = _lines(_small_logs(4).sender_text)
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER) + _crlf(lines[:-1]) + lines[-1][:5])
    for run in ("one", "two"):
        (tmp_path / run).mkdir()
        assert ex.main([str(raw), "--out", str(tmp_path / run / "s.csv")]) == 0
    for name in ("s.csv", "s.csv.audit.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_audit_records_hashes_and_no_local_paths(tmp_path):
    lines = _lines(_small_logs(5).sender_text)
    data = _crlf(BANNER) + _crlf(lines)
    raw = _write(tmp_path / "raw.txt", data)
    audit = ex.extract_file(raw, out=tmp_path / "s.csv")
    import hashlib
    assert audit["schema"] == "zephyr-echo-extract-audit-1"
    assert audit["raw_sha256"] == hashlib.sha256(data).hexdigest()
    assert audit["raw_bytes"] == len(data)
    block = audit["blocks"][0]
    assert block["output_sha256"] == hashlib.sha256((tmp_path / "s.csv").read_bytes()).hexdigest()
    assert block["first_raw_line"] == len(BANNER) + 1
    assert block["last_raw_line"] == len(BANNER) + len(lines)
    assert str(tmp_path) not in json.dumps(audit)


def test_cli_errors_exit_2_and_refuse_to_overwrite_the_raw_capture(tmp_path, capsys):
    raw = _write(tmp_path / "raw.txt", _crlf(BANNER))
    assert ex.main([str(raw), "--out", str(tmp_path / "s.csv")]) == 2
    assert "error:" in capsys.readouterr().err
    good = _write(tmp_path / "good.txt", _crlf(_lines(_small_logs(6).sender_text)))
    assert ex.main([str(good), "--out", str(good)]) == 2
    assert good.read_bytes() == _crlf(_lines(_small_logs(6).sender_text))
    assert ex.main([str(good), "--split", "--out", str(tmp_path / "x.csv")]) == 2
    assert ex.main([str(good), "--out-dir", str(tmp_path / "d")]) == 2


def test_a_second_carriage_return_at_a_line_end_fails_instead_of_reaching_the_output(tmp_path):
    lines = _lines(_small_logs(7).sender_text)
    data = _crlf(BANNER) + _crlf(lines[:18]) + lines[18] + b"\r\r\n" + _crlf(lines[19:])
    raw = _write(tmp_path / "raw.txt", data)
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{len(BANNER) + 19}: .*carriage return"):
        ex.extract_file(raw, out=tmp_path / "out.csv")


@pytest.mark.parametrize("data,line,message", [
    (b"\xef\xbb\xbf# zephyr-espnow-echo-log-1\n# role=sender\n", 1, "extra characters"),
    (b"boot\r# zephyr-espnow-echo-log-1\r# role=sender\r", 1, "extra characters"),
    (b"# zephyr-espnow-echo-log-1\n# board=A\nseq,tx_us,echo_us\n0,1,2\n", 1, "no # role= line"),
])
def test_bom_cr_only_endings_and_missing_role_fail(tmp_path, data, line, message):
    raw = _write(tmp_path / "raw.txt", data)
    with pytest.raises(ex.ExtractError, match=rf"raw.txt:{line}: .*{message}"):
        ex.extract_file(raw, out=tmp_path / "out.csv")


def test_role_is_read_from_its_own_block_only(tmp_path):
    second = _lines(_small_logs(2, n=10).sender_text)
    data = b"# zephyr-espnow-echo-log-1\n" + _crlf(second)
    raw = _write(tmp_path / "raw.txt", data)
    with pytest.raises(ex.ExtractError, match=r"raw.txt:1: .*no # role= line"):
        ex.extract_file(raw, out_dir=tmp_path / "blocks", split=True)


def test_output_paths_must_differ_and_must_not_exist(tmp_path):
    raw = _write(tmp_path / "raw.txt", _crlf(_lines(_small_logs(8).sender_text)))
    with pytest.raises(ex.ExtractError, match="same file"):
        ex.extract_file(raw, out=tmp_path / "s.csv", audit_path=tmp_path / "s.csv")
    with pytest.raises(ex.ExtractError, match="same file"):
        ex.extract_file(raw, out_dir=tmp_path / "d", split=True, audit_path=tmp_path / "d" / "block-1.csv")
    assert not (tmp_path / "s.csv").exists() and not (tmp_path / "d").exists()
    keep = _write(tmp_path / "keep.csv", b"precious\n")
    with pytest.raises(ex.ExtractError, match="already exists"):
        ex.extract_file(raw, out=keep)
    assert keep.read_bytes() == b"precious\n"


def test_a_bad_audit_path_leaves_no_log_behind(tmp_path):
    raw = _write(tmp_path / "raw.txt", _crlf(_lines(_small_logs(9).sender_text)))
    out = tmp_path / "s.csv"
    with pytest.raises(ex.ExtractError, match="folder"):
        ex.extract_file(raw, out=out, audit_path=tmp_path / "missing" / "a.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.txt"]


def _disk_ignores_case(folder):
    probe = folder / "CaseProbe"
    probe.write_bytes(b"")
    try:
        return (folder / "caseprobe").exists()
    finally:
        probe.unlink()


def test_case_variant_names_cannot_overwrite_the_log(tmp_path):
    # On a case-insensitive disk (default macOS, Windows) Case.csv and case.csv are one file.
    if not _disk_ignores_case(tmp_path):
        pytest.skip("this disk tells Case.csv and case.csv apart, so the collision cannot happen")
    raw = _write(tmp_path / "raw.txt", _crlf(_lines(_small_logs(10).sender_text)))
    with pytest.raises(ex.ExtractError, match="same file"):
        ex.extract_file(raw, out=tmp_path / "Case.csv", audit_path=tmp_path / "case.csv")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.txt"]


def test_case_variant_audit_name_in_split_folder_leaves_nothing_behind(tmp_path):
    if not _disk_ignores_case(tmp_path):
        pytest.skip("this disk tells block-1.csv and BLOCK-1.csv apart, so the collision cannot happen")
    raw = _write(tmp_path / "raw.txt", _crlf(_lines(_small_logs(11).sender_text)))
    folder = tmp_path / "d"
    with pytest.raises(ex.ExtractError, match="same file"):
        ex.extract_file(raw, out_dir=folder, split=True, audit_path=folder / "BLOCK-1.csv")
    assert not folder.exists() or list(folder.iterdir()) == []


def test_a_write_that_finds_an_existing_file_removes_what_it_created(tmp_path, monkeypatch):
    # Disk-independent: the audit write loses a race (file appears after the pre-checks).
    raw = _write(tmp_path / "raw.txt", _crlf(_lines(_small_logs(12).sender_text)))
    real = ex._write_text
    def racing(path, text):
        if path.name == "a.json":
            raise FileExistsError(path)
        real(path, text)
    monkeypatch.setattr(ex, "_write_text", racing)
    with pytest.raises(ex.ExtractError, match="same file|already exists"):
        ex.extract_file(raw, out=tmp_path / "s.csv", audit_path=tmp_path / "a.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw.txt"]
