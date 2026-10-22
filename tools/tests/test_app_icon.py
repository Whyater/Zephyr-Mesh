"""Checks for the checked-in app icon files, read with the standard library only.

These parse the files independently of the tools that wrote them: the .icns
chunk list, the .ico directory, and each PNG's IHDR header.
"""
from __future__ import annotations

from pathlib import Path
import struct

import pytest

from tools.pack_ico import pack_ico, png_size

ROOT = Path(__file__).parents[2]
ICON = ROOT / "tools" / "design" / "icon"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Apple's icns element types and their pixel sizes. iconutil stores 16 and 32 px
# as ARGB data (ic04, ic05) and every other size as PNG.
ICNS_PIXELS = {
    "ic04": 16, "ic05": 32, "ic11": 32, "ic12": 64, "ic07": 128, "ic13": 256,
    "ic08": 256, "ic14": 512, "ic09": 512, "ic10": 1024,
}


def ihdr(data: bytes) -> tuple[int, int]:
    """Width and height straight from the IHDR chunk."""
    assert data[:8] == PNG_SIGNATURE, "missing PNG signature"
    length, kind = struct.unpack(">I4s", data[8:16])
    assert kind == b"IHDR" and length == 13
    return struct.unpack(">II", data[16:24])


def icns_elements(data: bytes) -> dict[str, bytes]:
    assert data[:4] == b"icns"
    assert struct.unpack(">I", data[4:8])[0] == len(data), "icns length field must match the file"
    elements: dict[str, bytes] = {}
    offset = 8
    while offset < len(data):
        kind, length = struct.unpack(">4sI", data[offset:offset + 8])
        assert length >= 8 and offset + length <= len(data), f"bad element length at {offset}"
        name = kind.decode("ascii")
        assert name not in elements, f"duplicate icns element {name}"
        elements[name] = data[offset + 8:offset + length]
        offset += length
    assert offset == len(data)
    return elements


@pytest.mark.parametrize("name", ["ZephyrMesh.icns", "ZephyrMesh.ico", "ZephyrMesh-256.png", "ZephyrMesh-1024.png"])
def test_icon_files_exist(name: str) -> None:
    path = ICON / name
    assert path.is_file() and path.stat().st_size > 0


@pytest.mark.parametrize(("name", "pixels"), [("ZephyrMesh-256.png", 256), ("ZephyrMesh-1024.png", 1024)])
def test_png_dimensions(name: str, pixels: int) -> None:
    assert ihdr((ICON / name).read_bytes()) == (pixels, pixels)


def test_icns_has_every_iconset_size() -> None:
    elements = icns_elements((ICON / "ZephyrMesh.icns").read_bytes())
    image_types = {kind for kind in elements if kind in ICNS_PIXELS}
    assert image_types == set(ICNS_PIXELS), f"icns image types: {sorted(elements)}"
    for kind, pixels in ICNS_PIXELS.items():
        payload = elements[kind]
        if payload[:8] == PNG_SIGNATURE:
            assert ihdr(payload) == (pixels, pixels), kind
        else:
            assert payload[:4] == b"ARGB", f"{kind} is neither PNG nor ARGB"
    assert {pixels for pixels in ICNS_PIXELS.values()} == {16, 32, 64, 128, 256, 512, 1024}


def test_ico_has_seven_png_frames() -> None:
    data = (ICON / "ZephyrMesh.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind, count) == (0, 1, 7)
    sizes = []
    for index in range(count):
        width, height, colors, zero, planes, bits, length, offset = struct.unpack(
            "<BBBBHHII", data[6 + 16 * index:22 + 16 * index])
        width, height = width or 256, height or 256
        assert (colors, zero, planes, bits) == (0, 0, 1, 32)
        assert ihdr(data[offset:offset + length]) == (width, height)
        assert width == height
        sizes.append(width)
    assert sizes == [16, 24, 32, 48, 64, 128, 256]


def test_pack_ico_rejects_bad_frames() -> None:
    def fake_png(width: int, height: int) -> bytes:
        return PNG_SIGNATURE + struct.pack(">I4sII", 13, b"IHDR", width, height) + b"\x08\x06\x00\x00\x00"

    assert png_size(fake_png(48, 48)) == (48, 48)
    with pytest.raises(ValueError):
        pack_ico([])
    with pytest.raises(ValueError):
        pack_ico([fake_png(32, 16)])
    with pytest.raises(ValueError):
        pack_ico([fake_png(512, 512)])
    with pytest.raises(ValueError):
        pack_ico([fake_png(32, 32), fake_png(32, 32)])
    with pytest.raises(ValueError):
        pack_ico([b"not a png"])
