"""Tests for tools/compare_reproduce_tolerance.py on small hand-made file pairs.

Each pair is built so the expected class is known by construction: a file with
CRLF line endings, a float that moved in its last digit, a count that changed,
and PNGs with equal or different pixels.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from sim.tests import platform_compare
from tools import compare_reproduce_tolerance as tool


def pair(tmp_path: Path, name: str, expected: bytes, actual: bytes) -> tuple[Path, Path]:
    (tmp_path / "e").mkdir(exist_ok=True)
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "e" / name).write_bytes(expected)
    (tmp_path / "a" / name).write_bytes(actual)
    return tmp_path / "e" / name, tmp_path / "a" / name


def test_identical_bytes(tmp_path):
    assert tool.classify(*pair(tmp_path, "x.json", b'{"a": 1}\n', b'{"a": 1}\n'))[0] == "byte-identical"


def test_crlf_only(tmp_path):
    cls, info = tool.classify(*pair(tmp_path, "r.json", b'{\n "a": 1\n}\n', b'{\r\n "a": 1\r\n}\r\n'))
    assert cls == "line-endings-only"
    assert info == {"crlf_expected": 0, "crlf_found": 3}


def test_last_digit_float_in_json(tmp_path):
    # 0.1 + 0.2 and 0.3 are neighbouring doubles: a relative gap of about 1.9e-16.
    cls, info = tool.classify(*pair(tmp_path, "s.json", b'{"v": 0.30000000000000004}', b'{"v": 0.3}'))
    assert cls == "numeric-within-tol"
    assert 1e-16 < info["max_relative_gap"] < 2e-16


def test_float_and_crlf_together(tmp_path):
    cls, info = tool.classify(*pair(tmp_path, "s.json", b'{"v": 2.0}\n', b'{"v": 2.0000000000000004}\r\n'))
    assert cls == "numeric-within-tol"
    assert info["line_endings_also_differ"] is True


def test_changed_count_is_real(tmp_path):
    # One more counted event is a real change, however small the float gaps elsewhere.
    cls, info = tool.classify(*pair(tmp_path, "run.json", b'{"entries": 4, "x": 1.0}', b'{"entries": 3, "x": 1.0}'))
    assert cls == "real-difference"
    assert info["first"] == ["$.entries", "4", "3"]
    assert info["max_relative_gap"] == pytest.approx(0.25)


def test_changed_whole_number_in_csv_or_text_is_real(tmp_path):
    # A seed or a count is compared exactly: a 1e-9 relative tolerance would let 3287529750 pass as
    # 3287529751, since the gap is only about 3e-10 of the value.
    for name, e, a in (("per_seed.csv", b"seed,x\n3287529750,1.0\n", b"seed,x\n3287529751,1.0\n"),
                       ("log.txt", b"seed 3287529750\n", b"seed 3287529751\n")):
        cls, _ = tool.classify(*pair(tmp_path, name, e, a))
        assert cls == "real-difference", name


def test_path_separator_is_real(tmp_path):
    cls, info = tool.classify(*pair(tmp_path, "run.json", b'{"cmd": "sim/runs/a"}', b'{"cmd": "sim\\\\runs\\\\a"}'))
    assert cls == "real-difference"
    assert info["first_non_number"][0] == "$.cmd"


def test_key_order_only(tmp_path):
    assert tool.classify(*pair(tmp_path, "k.json", b'{"a": 1, "b": 2}', b'{"b": 2, "a": 1}'))[0] == "formatting-only"


def test_csv_last_digit_and_real_change(tmp_path):
    e, a = pair(tmp_path, "t.csv", b"x,y\r\n1,0.30000000000000004\r\n", b"x,y\r\n1,0.3\r\n")
    assert tool.classify(e, a)[0] == "numeric-within-tol"
    e, a = pair(tmp_path, "u.csv", b"x,y\r\n1,1.5\r\n", b"x,y\r\n1,1.6\r\n")
    cls, info = tool.classify(e, a)
    assert cls == "real-difference"
    assert info["first"] == {"line": 2, "expected": "1.5", "found": "1.6"}


def test_csv_label_change_is_real(tmp_path):
    cls, info = tool.classify(*pair(tmp_path, "l.csv", b"k,v\r\nok,1\r\n", b"k,v\r\nfail,1\r\n"))
    assert cls == "real-difference"
    assert info["first_non_number"]["expected"] == "ok"


def test_gzip_csv(tmp_path):
    e = gzip.compress(b"v\r\n0.30000000000000004\r\n", mtime=0)
    a = gzip.compress(b"v\r\n0.3\r\n", mtime=1)
    assert tool.classify(*pair(tmp_path, "p.csv.gz", e, a))[0] == "numeric-within-tol"


def test_png_pixels(tmp_path):
    import matplotlib.image as mpimg

    image = np.zeros((4, 5, 3))
    image[1, 2] = (1.0, 0.5, 0.0)
    (tmp_path / "e").mkdir()
    (tmp_path / "a").mkdir()
    mpimg.imsave(tmp_path / "e" / "f.png", image, metadata={"Software": "one"})
    mpimg.imsave(tmp_path / "a" / "f.png", image, metadata={"Software": "two"})
    assert tool.classify(tmp_path / "e" / "f.png", tmp_path / "a" / "f.png")[0] == "pixels-identical"
    image[0, 0] = (0.0, 0.0, 1.0)
    mpimg.imsave(tmp_path / "a" / "f.png", image)
    cls, info = tool.classify(tmp_path / "e" / "f.png", tmp_path / "a" / "f.png")
    assert cls == "real-difference"
    assert info["pixels_differing"] == 1


def test_tolerance_rule_restores_platform_flag(tmp_path):
    before = platform_compare.EXACT
    tool.classify(*pair(tmp_path, "s.json", b'{"v": 0.30000000000000004}', b'{"v": 0.3}'))
    assert platform_compare.EXACT is before


def test_main_exit_code_and_output(tmp_path, capsys):
    pair(tmp_path, "a.json", b'{"v": 1}', b'{"v": 1}')
    pair(tmp_path, "b.json", b'{"n": 1}', b'{"n": 2}')
    assert tool.main([str(tmp_path / "e"), str(tmp_path / "a")]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["file"], r["class"]) for r in lines] == [("a.json", "byte-identical"), ("b.json", "real-difference")]
    assert tool.main([str(tmp_path / "e"), str(tmp_path / "a"), "a.json"]) == 0
