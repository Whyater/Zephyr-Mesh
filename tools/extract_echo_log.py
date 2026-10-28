"""Cut the echo log block out of a raw serial capture.

A board's USB serial port prints more than the log: boot messages before the
app starts, ``\\r\\n`` line endings, and, if the board resets mid-run, a second
set of boot messages and a second log. The importer
(``tools/import_espnow_log.py``) rejects all of that on purpose, so this tool
sits in front of it and writes a clean ``# zephyr-espnow-echo-log-1`` file
plus an audit JSON that lists every line it left out.

Rules, in the order they are applied:

1. The capture is split on ``\\n``. One ``\\r`` at the end of a line is dropped
   (counted as ``crlf_lines``). Any other ``\\r`` inside a log block fails.
2. If the capture does not end in ``\\n``, its last piece was cut off
   mid-line. It is dropped and reported as ``truncated_final_line``.
3. Everything before the first line that is exactly the format line is boot
   text. It is dropped and listed (reason ``before_first_block``). Boot text
   may hold bytes that are not UTF-8; the audit shows them as ``\\xNN``.
4. A line that contains the format line with anything else on it fails, and
   so does a capture with no format line at all.
5. A second format line means the board reset mid-run. That fails unless
   ``--split`` is given. With ``--split``, the lines after a block's last
   row-shaped line (digits and commas with the right number of fields) and
   before the next format line are the reset's boot text: they are dropped
   and listed (reason ``reset_boot_text``). A board can reset in the middle
   of printing a row, so that block's last row may be cut short and still
   look whole; the audit marks it ``last_row_may_be_cut``.
6. Every kept block is checked line by line with the importer's own parser,
   so anything else inside a block (a Wi-Fi warning, a panic message, a blank
   line, bytes that are not UTF-8) fails on its line number in the raw
   capture. Nothing inside a block is ever skipped or repaired.

Nothing is written unless every block passes and every output path is new,
distinct, and in a folder that exists (``--split`` makes its own folder). The audit records the raw
capture's SHA-256 and each output's SHA-256, and uses file names only, never
folders. It does not say whether the capture is a measurement; the importer's
``--status`` sets that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.import_espnow_log import (
    FORMAT_LINE,
    EchoLogError,
    parse_echoer_log,
    parse_sender_log,
)

SCHEMA = "zephyr-echo-extract-audit-1"
_FORMAT = FORMAT_LINE.encode("ascii")
_PARSERS = {"sender": (parse_sender_log, 3), "echoer": (parse_echoer_log, 4)}
_ROW_FIELD = re.compile(rb"-?[0-9]*")
_BLOCK_LABEL = "block"
_BLOCK_ERROR = re.compile(rf"{_BLOCK_LABEL}:(\d+): (.*)", re.DOTALL)


class ExtractError(ValueError):
    """A capture that cannot be cut cleanly, with the raw file name and line."""


def _error(name: str, line: int, message: str) -> ExtractError:
    return ExtractError(f"{name}:{line}: {message}")


def _shown(raw: bytes) -> str:
    """A line as text for the audit, with any non-UTF-8 byte written as \\xNN."""
    return raw.decode("utf-8", errors="backslashreplace")


def _split_lines(data: bytes) -> tuple[list[bytes], bytes, int]:
    """Complete lines (one trailing \\r dropped), the unfinished tail, and the \\r count."""
    pieces = data.split(b"\n")
    tail = pieces.pop()
    lines: list[bytes] = []
    crlf = 0
    for piece in pieces:
        if piece.endswith(b"\r"):
            piece = piece[:-1]
            crlf += 1
        lines.append(piece)
    return lines, tail, crlf


def _role(lines: list[bytes], start: int, end: int, name: str) -> str:
    """The block's role from its ``# role=`` header line (looking only inside the block)."""
    index = start + 1
    while index < end and lines[index].startswith(b"# "):
        if lines[index].startswith(b"# role="):
            value = lines[index][len(b"# role="):]
            if value in (b"sender", b"echoer"):
                return value.decode("ascii")
            raise _error(name, index + 1, "role must be sender or echoer")
        index += 1
    raise _error(name, start + 1, "this log block has no # role= line in its header")


def _row_shaped(line: bytes, n_fields: int) -> bool:
    fields = line.split(b",")
    return len(fields) == n_fields and all(_ROW_FIELD.fullmatch(field) for field in fields)


def _check_block(lines: list[bytes], start: int, end: int, role: str, name: str) -> str:
    """Decode a block strictly and run the importer's parser on it; return its text."""
    decoded: list[str] = []
    for index in range(start, end):
        if b"\r" in lines[index]:
            raise _error(name, index + 1, "stray carriage return inside the log block")
        try:
            decoded.append(lines[index].decode("utf-8"))
        except UnicodeDecodeError:
            raise _error(name, index + 1, "this line inside the log block is not valid UTF-8") from None
    text = "\n".join(decoded) + "\n"
    parser = _PARSERS[role][0]
    try:
        parser(text, _BLOCK_LABEL)
    except EchoLogError as exc:
        match = _BLOCK_ERROR.fullmatch(str(exc))
        if match is None:
            raise ExtractError(f"{name}: {exc}") from None
        raise _error(name, start + int(match.group(1)), match.group(2)) from None
    return text


def extract(data: bytes, name: str, *, split: bool = False) -> tuple[list[str], dict[str, Any]]:
    """Cut the log block(s) out of raw capture bytes.

    Returns the clean text of each block and the audit (without output names
    or hashes, which the writer adds). ``name`` labels error messages.
    """
    if data == b"":
        raise _error(name, 1, "empty capture")
    lines, tail, crlf = _split_lines(data)
    starts: list[int] = []
    for index, line in enumerate(lines):
        if line == _FORMAT:
            starts.append(index)
        elif _FORMAT in line:
            raise _error(name, index + 1, f"this line holds {FORMAT_LINE} with extra characters on it")
    if not starts and _FORMAT in tail:
        raise _error(name, len(lines) + 1, f"this line holds {FORMAT_LINE} with extra characters on it "
                                           "(no \\n line endings?)")
    if not starts:
        raise _error(name, 1, f"no line is exactly {FORMAT_LINE}; is this an echo log capture?")
    if len(starts) > 1 and not split:
        raise _error(name, starts[1] + 1, "a second format line (did the board reset mid-run?); "
                                          "use --split to write one file per block")

    dropped: list[dict[str, Any]] = [
        {"raw_line": index + 1, "reason": "before_first_block", "text": _shown(lines[index])}
        for index in range(starts[0])]
    texts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for k, start in enumerate(starts):
        is_last = k == len(starts) - 1
        end = len(lines) if is_last else starts[k + 1]
        role = _role(lines, start, end, name)
        keep_end = end
        if not is_last:
            n_fields = _PARSERS[role][1]
            shaped = [index for index in range(start + 1, end) if _row_shaped(lines[index], n_fields)]
            if shaped:
                keep_end = shaped[-1] + 1
            dropped += [{"raw_line": index + 1, "reason": "reset_boot_text", "text": _shown(lines[index])}
                        for index in range(keep_end, end)]
        text = _check_block(lines, start, keep_end, role, name)
        header_lines = sum(1 for index in range(start, keep_end) if lines[index].startswith(b"# "))
        texts.append(text)
        blocks.append({
            "index": k + 1,
            "role": role,
            "first_raw_line": start + 1,
            "last_raw_line": keep_end,
            "data_rows": keep_end - start - header_lines - 1,
            "ended_by": "end_of_capture" if is_last else "reset",
            "last_row_may_be_cut": not is_last,
        })

    truncated = None
    if tail:
        truncated = {"raw_line": len(lines) + 1, "text": _shown(tail)}
        dropped.append({"raw_line": len(lines) + 1, "reason": "truncated_final_line", "text": _shown(tail)})
    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "raw_name": name,
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "raw_bytes": len(data),
        "raw_lines": len(lines) + (1 if tail else 0),
        "crlf_lines": crlf,
        "truncated_final_line": truncated,
        "blocks": blocks,
        "dropped": dropped,
        "note": ("Lines left out of the capture, by reason. Every kept block passed the importer's parser. "
                 "This audit does not say whether the capture is a measurement; the importer's --status sets that."),
    }
    return texts, audit


def _write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def extract_file(raw_path: str | Path, *, out: str | Path | None = None, out_dir: str | Path | None = None,
                 split: bool = False, audit_path: str | Path | None = None) -> dict[str, Any]:
    """Read a raw capture, write the clean log(s) and the audit, and return the audit.

    Without ``split``, ``out`` is the clean log and the audit defaults to
    ``<out>.audit.json``. With ``split``, ``out_dir`` (empty or new) gets
    ``block-1.csv``, ``block-2.csv``, ... and the audit defaults to
    ``<out_dir>/audit.json``.
    """
    raw_path = Path(raw_path)
    if split and (out_dir is None or out is not None):
        raise ExtractError("--split writes into --out-dir; do not give --out")
    if not split and (out is None or out_dir is not None):
        raise ExtractError("give --out for one block (or --split with --out-dir for several)")
    texts, audit = extract(raw_path.read_bytes(), raw_path.name, split=split)

    if split:
        folder = Path(out_dir)
        if folder.exists() and any(folder.iterdir()):
            raise ExtractError(f"{folder.name} is not empty; pick a new or empty folder")
        targets = [folder / f"block-{k}.csv" for k in range(1, len(texts) + 1)]
        audit_target = Path(audit_path) if audit_path is not None else folder / "audit.json"
    else:
        targets = [Path(out)]
        audit_target = Path(audit_path) if audit_path is not None else Path(str(out) + ".audit.json")
    resolved = [target.resolve() for target in [*targets, audit_target]]
    if raw_path.resolve() in resolved:
        raise ExtractError(f"refusing to overwrite the raw capture {raw_path.name}")
    if len(set(resolved)) != len(resolved):
        raise ExtractError("two outputs would be the same file; give --audit a different name")
    for target in [*targets, audit_target]:
        if target.exists():
            raise ExtractError(f"{target.name} already exists; pick a new name")
        if not (split and target.parent == Path(out_dir)) and not target.parent.is_dir():
            raise ExtractError(f"the folder for {target.name} does not exist")

    for block, target, text in zip(audit["blocks"], targets, texts):
        block["output"] = target.name
        block["output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    audit_text = json.dumps(audit, indent=2, allow_nan=False) + "\n"
    new_folders: list[Path] = []
    if split:
        folder = Path(out_dir)
        for ancestor in [folder, *folder.parents]:
            if ancestor.exists():
                break
            new_folders.append(ancestor)
        folder.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        # Exclusive create: on a disk that ignores case, Case.csv and case.csv are one file, and
        # the second write must fail instead of replacing the first.
        for target, text in [*zip(targets, texts), (audit_target, audit_text)]:
            created.append(target)
            _write_text(target, text)
    except BaseException as exc:
        # Leave nothing half-written. The file that raised FileExistsError is not ours to delete.
        for path in created[:-1] if isinstance(exc, FileExistsError) else created:
            path.unlink(missing_ok=True)
        for path in new_folders:
            try:
                path.rmdir()
            except OSError:
                break
        if isinstance(exc, FileExistsError):
            raise ExtractError(f"{created[-1].name} already exists, or two outputs are the same file; "
                               "pick new names") from exc
        raise
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut the zephyr-espnow-echo-log-1 block out of a raw serial capture.")
    parser.add_argument("raw", help="raw capture file (bytes exactly as the serial port sent them)")
    parser.add_argument("--out", help="clean log file to write (one block)")
    parser.add_argument("--split", action="store_true", help="allow several blocks (board resets); needs --out-dir")
    parser.add_argument("--out-dir", help="empty or new folder for block-1.csv, block-2.csv, ... with --split")
    parser.add_argument("--audit", help="audit JSON path (default <out>.audit.json or <out-dir>/audit.json)")
    args = parser.parse_args(argv)
    try:
        audit = extract_file(args.raw, out=args.out, out_dir=args.out_dir, split=args.split, audit_path=args.audit)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = sum(block["data_rows"] for block in audit["blocks"])
    print(f"{len(audit['blocks'])} block(s), {rows} data rows, {len(audit['dropped'])} line(s) left out")
    if audit["truncated_final_line"] is not None:
        print(f"warning: line {audit['truncated_final_line']['raw_line']} was cut off mid-line and left out",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
