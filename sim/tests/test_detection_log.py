import dataclasses
import json
from pathlib import Path

import pytest

from sim.detection_log import (
    DETECTION_LOG_SCHEMA,
    DetectionLogError,
    detection_log_lines,
    parse_detection_log_lines,
    read_detection_log,
    write_detection_log,
)
from sim.vision_trace import VisionTrace

SIM_DIR = Path(__file__).resolve().parents[2] / "sim"
EXAMPLE_LOG = SIM_DIR / "detection_log_example.jsonl"
EXAMPLE_TRACE = SIM_DIR / "vision_trace_example.json"


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def header(**changes):
    value = {
        "schema": DETECTION_LOG_SCHEMA,
        "trace_id": "log-test",
        "status": "fixture",
        "clock_domain": "shared_monotonic",
        "capture": {"source_type": "camera", "source_id": "cam", "device_ids": ["host"], "model": "m"},
    }
    value.update(changes)
    return value


def detected(index, **changes):
    value = {
        "observation_id": f"obs-{index}",
        "frame_id": f"frame-{index}",
        "subject_id": "s1",
        "capture_time_s": 1.0 + index / 10,
        "processing_time_s": 1.01 + index / 10,
        "outcome": "detected",
        "confidence": 0.9,
        "position_m": [0.1, 0.2, 2.0],
        "position_frame": "camera",
    }
    value.update(changes)
    return value


def write_lines(tmp_path, lines, name="log.jsonl"):
    path = tmp_path / name
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def expect_error(path, line, fragment=None):
    with pytest.raises(DetectionLogError) as info:
        read_detection_log(path)
    message = str(info.value)
    assert message.startswith(f"{path}:{line}: "), message
    if fragment is not None:
        assert fragment in message
    return message


def test_example_log_reads():
    trace = read_detection_log(EXAMPLE_LOG)
    assert trace.status == "fixture"
    assert trace.clock_domain == "shared_monotonic"
    assert trace.notes and "fixture" in trace.notes
    outcomes = [item.outcome for item in trace.observations]
    assert 8 <= len(outcomes) <= 12
    assert {"detected", "missed", "unknown"} <= set(outcomes)
    assert {item.subject_id for item in trace.observations} == {"subject-001"}
    for item in trace.observations:
        if item.outcome == "detected":
            assert item.position_frame == "camera"
            assert item.ground_truth is not None


def test_example_log_file_format():
    raw = EXAMPLE_LOG.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw.split(b"\n")[0])["schema"] == DETECTION_LOG_SCHEMA


def test_round_trip_example_log(tmp_path):
    original = read_detection_log(EXAMPLE_LOG)
    out = tmp_path / "copy.jsonl"
    write_detection_log(original, out)
    assert read_detection_log(out) == original
    assert out.read_bytes().endswith(b"\n")
    assert b"\r" not in out.read_bytes()


def test_round_trip_example_trace(tmp_path):
    original = VisionTrace.from_path(EXAMPLE_TRACE)
    out = tmp_path / "trace.jsonl"
    write_detection_log(original, out)
    again = read_detection_log(out)
    assert again == original
    assert len(out.read_bytes().splitlines()) == 1 + len(original.observations)


def test_written_lines_are_compact_and_sorted():
    trace = read_detection_log(EXAMPLE_LOG)
    lines = detection_log_lines(trace)
    assert len(lines) == 1 + len(trace.observations)
    for text in lines:
        value = json.loads(text)
        assert text == json.dumps(value, sort_keys=True, separators=(",", ":"))
        assert list(value) == sorted(value)
    assert json.loads(lines[0])["schema"] == DETECTION_LOG_SCHEMA
    assert "observations" not in json.loads(lines[0])
    assert "bbox_px" not in json.loads(lines[1])


def test_parse_lines_accepts_terminated_and_bare_lines():
    lines = [dump(header()), dump(detected(1))]
    bare = parse_detection_log_lines(lines)
    terminated = parse_detection_log_lines([line + "\n" for line in lines])
    assert bare == terminated
    assert len(bare.observations) == 1


def test_parse_lines_error_uses_source_name():
    with pytest.raises(DetectionLogError, match=r"^my-source:2: "):
        parse_detection_log_lines([dump(header()), "{"], source_name="my-source")


def test_header_only_log(tmp_path):
    path = write_lines(tmp_path, [dump(header())])
    trace = read_detection_log(path)
    assert trace.observations == ()
    out = tmp_path / "again.jsonl"
    write_detection_log(trace, out)
    assert read_detection_log(out) == trace


def test_no_final_newline_is_accepted(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes("\n".join([dump(header()), dump(detected(1))]).encode("utf-8"))
    assert len(read_detection_log(path).observations) == 1


def test_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    expect_error(path, 0, "empty")


def test_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "bad.jsonl"
    good = dump(header()).encode("utf-8")
    bad = dump(detected(1)).encode("utf-8").replace(b"obs-1", b"obs-\xff")
    path.write_bytes(good + b"\n" + bad + b"\n")
    expect_error(path, 2, "UTF-8")


def test_rejects_invalid_json(tmp_path):
    path = write_lines(tmp_path, [dump(header()), dump(detected(1)), '{"observation_id": '])
    expect_error(path, 3, "JSON")


def test_rejects_json_that_is_not_an_object(tmp_path):
    path = write_lines(tmp_path, [dump(header()), "[1, 2, 3]"])
    expect_error(path, 2, "object")
    path = write_lines(tmp_path, ['"just text"'], name="first.jsonl")
    expect_error(path, 1, "object")


def test_rejects_duplicate_keys(tmp_path):
    good = dump(detected(1))
    duplicated = good[:-1] + ',"outcome":"missed"}'
    path = write_lines(tmp_path, [dump(header()), good.replace("obs-1", "obs-0"), duplicated])
    expect_error(path, 3, "duplicate key")


def test_rejects_duplicate_keys_in_nested_object(tmp_path):
    line = dump(header()).replace('"source_id":"cam"', '"source_id":"cam","source_id":"cam2"')
    path = write_lines(tmp_path, [line])
    expect_error(path, 1, "duplicate key")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_finite_tokens(tmp_path, token):
    line = dump(detected(1)).replace('"confidence":0.9', f'"confidence":{token}')
    assert token in line
    path = write_lines(tmp_path, [dump(header()), line])
    expect_error(path, 2, token)


def test_rejects_overflowing_number(tmp_path):
    line = dump(detected(1)).replace('"capture_time_s":1.1', '"capture_time_s":1e999')
    assert "1e999" in line
    path = write_lines(tmp_path, [dump(header()), line])
    expect_error(path, 2, "finite")


def test_rejects_missing_header(tmp_path):
    path = write_lines(tmp_path, [dump(detected(1))])
    expect_error(path, 1, "header")


def test_rejects_wrong_schema(tmp_path):
    path = write_lines(tmp_path, [dump(header(schema="zephyr-vision-trace-1"))])
    expect_error(path, 1, DETECTION_LOG_SCHEMA)


def test_rejects_observations_inside_header(tmp_path):
    path = write_lines(tmp_path, [dump(header(observations=[]))])
    expect_error(path, 1, "observations")


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"status": "guess"}, "status"),
        ({"clock_domain": "wall"}, "clock_domain"),
        ({"capture": {"source_type": "camera"}}, "capture"),
        ({"trace_id": ""}, "trace_id"),
        ({"surprise": 1}, "unknown field"),
        ({"status": "measured"}, "measured"),
    ],
)
def test_rejects_bad_manifest_on_line_one(tmp_path, changes, fragment):
    path = write_lines(tmp_path, [dump(header(**changes)), dump(detected(1))])
    message = expect_error(path, 1, fragment)
    assert "header" in message


def test_rejects_bad_observation(tmp_path):
    bad = detected(2, confidence=1.5)
    path = write_lines(tmp_path, [dump(header()), dump(detected(1)), dump(bad)])
    message = expect_error(path, 3, "confidence")
    assert "observation" in message


def test_rejects_observation_with_unknown_field(tmp_path):
    path = write_lines(tmp_path, [dump(header()), dump(detected(1, extra=1))])
    expect_error(path, 2, "unknown field")


def test_rejects_missed_without_reason(tmp_path):
    missed = {"observation_id": "m1", "frame_id": "f", "subject_id": "s1", "capture_time_s": 1.0, "outcome": "missed"}
    path = write_lines(tmp_path, [dump(header()), dump(missed)])
    expect_error(path, 2, "miss_reason")


def test_rejects_duplicate_observation_id_at_second_line(tmp_path):
    lines = [dump(header()), dump(detected(1)), dump(detected(2)), dump(detected(1, frame_id="other"))]
    path = write_lines(tmp_path, lines)
    message = expect_error(path, 4, "duplicate observation_id")
    assert "line 2" in message


def test_rejects_blank_line_in_middle(tmp_path):
    path = write_lines(tmp_path, [dump(header()), dump(detected(1)), "", dump(detected(2))])
    expect_error(path, 3, "blank")


def test_rejects_whitespace_only_line(tmp_path):
    path = write_lines(tmp_path, [dump(header()), "   \t", dump(detected(1))])
    expect_error(path, 2, "blank")


def test_rejects_second_trailing_newline(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes((dump(header()) + "\n" + dump(detected(1)) + "\n\n").encode("utf-8"))
    expect_error(path, 3, "blank")


def test_rejects_blank_header_line(tmp_path):
    path = write_lines(tmp_path, ["", dump(header())])
    expect_error(path, 1, "blank")


def test_rejects_crlf_files(tmp_path):
    path = tmp_path / "crlf.jsonl"
    path.write_bytes((dump(header()) + "\r\n" + dump(detected(1)) + "\r\n").encode("utf-8"))
    message = expect_error(path, 1, "carriage return")
    assert "CRLF" in message


def test_rejects_cr_inside_later_line(tmp_path):
    path = tmp_path / "cr.jsonl"
    path.write_bytes((dump(header()) + "\n" + dump(detected(1)) + "\n" + dump(detected(2)) + "\r\n").encode("utf-8"))
    expect_error(path, 3, "carriage return")


def test_rejects_shared_clock_violation_with_line(tmp_path):
    early = detected(2, processing_time_s=0.5)
    path = write_lines(tmp_path, [dump(header()), dump(detected(1)), dump(early)])
    expect_error(path, 3, "shared_monotonic")


def test_other_clock_domains_allow_early_processing_time(tmp_path):
    early = detected(1, processing_time_s=0.5)
    path = write_lines(tmp_path, [dump(header(clock_domain="device_and_host")), dump(early)])
    assert read_detection_log(path).observations[0].processing_time_s == 0.5


def test_first_problem_wins(tmp_path):
    path = write_lines(tmp_path, [dump(header()), dump(detected(1, confidence=9)), "{", ""])
    expect_error(path, 2, "confidence")


def test_non_ascii_text_round_trips(tmp_path):
    trace = parse_detection_log_lines([dump(header(notes="caf\u00e9 fixture")), dump(detected(1))])
    out = tmp_path / "unicode.jsonl"
    write_detection_log(trace, out)
    assert read_detection_log(out) == trace
    assert read_detection_log(out).notes == "caf\u00e9 fixture"


def test_rejects_huge_integer_with_line(tmp_path):
    # Python parses a 1001-digit integer, but turning it into a float overflows.
    line = dump(detected(1)).replace('"capture_time_s":1.1', '"capture_time_s":' + "9" * 1001)
    path = write_lines(tmp_path, [dump(header()), line])
    expect_error(path, 2, "observation:")


def test_rejects_huge_integer_in_header_with_line(tmp_path):
    path = write_lines(tmp_path, [dump(header(clock_uncertainty_s=int("9" * 1001)))])
    expect_error(path, 1, "header:")


def test_rejects_utf8_bom_on_line_one(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + (dump(header()) + "\n").encode("utf-8"))
    message = expect_error(path, 1, "byte order mark (BOM)")
    assert "invalid JSON" not in message


def test_rejects_non_object_ground_truth_with_line(tmp_path):
    line = dump(detected(1, ground_truth=[10.0, -5.0, 2.0]))
    path = write_lines(tmp_path, [dump(header()), line])
    expect_error(path, 2, "ground_truth must be an object")


def test_write_rejects_non_finite_header_value(tmp_path):
    trace = parse_detection_log_lines([dump(header()), dump(detected(1))])
    bad = dataclasses.replace(trace, provenance={"instrument": {"kind": "k", "model": "m", "settings": {"gain": float("nan")}}})
    path = tmp_path / "out.jsonl"
    with pytest.raises(DetectionLogError) as info:
        write_detection_log(bad, path)
    assert str(info.value).startswith(f"{path}:1: cannot write header: ")
    assert not path.exists()


def test_write_rejects_non_finite_observation_value(tmp_path):
    trace = parse_detection_log_lines([dump(header()), dump(detected(1)), dump(detected(2))])
    second = dataclasses.replace(trace.observations[1], confidence=float("inf"))
    bad = dataclasses.replace(trace, observations=(trace.observations[0], second))
    with pytest.raises(DetectionLogError, match=r"^<memory>:3: cannot write observation 'obs-2': "):
        detection_log_lines(bad)
