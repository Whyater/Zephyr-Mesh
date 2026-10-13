#!/usr/bin/env python3
"""Pack square PNG frames into one Windows .ico file, standard library only.

An .ico file is a 6-byte header, one 16-byte directory entry per frame, then
the frame data. Windows Vista and later accept PNG data inside an entry, so
each frame is stored as the PNG bytes unchanged.

Usage: python3 tools/pack_ico.py OUT.ico FRAME.png [FRAME.png ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) from a PNG's IHDR chunk."""
    if data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file with an IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def pack_ico(frames: list[bytes]) -> bytes:
    """Build .ico bytes from PNG frames, smallest first."""
    if not frames:
        raise ValueError("an .ico needs at least one frame")
    sized = []
    for data in frames:
        width, height = png_size(data)
        if width != height or not 1 <= width <= 256:
            raise ValueError(f"frames must be square and at most 256 px, got {width}x{height}")
        sized.append((width, data))
    sized.sort(key=lambda item: item[0])
    if len({width for width, _ in sized}) != len(sized):
        raise ValueError("each frame size may appear only once")
    header = struct.pack("<HHH", 0, 1, len(sized))
    offset = len(header) + 16 * len(sized)
    entries = b""
    payload = b""
    for width, data in sized:
        stored = 0 if width == 256 else width  # 0 means 256 in the directory entry
        entries += struct.pack("<BBBBHHII", stored, stored, 0, 0, 1, 32, len(data), offset)
        payload += data
        offset += len(data)
    return header + entries + payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("frames", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        data = pack_ico([path.read_bytes() for path in args.frames])
    except (OSError, ValueError) as exc:
        print(f"pack_ico: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
