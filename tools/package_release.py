#!/usr/bin/env python3
"""Package one built desktop application as a deterministic release ZIP.

The updater expects one application directory at the root of each archive.
This script keeps that shape explicit, rejects symlinks, fixes ZIP metadata for
stable output, and writes a conventional SHA-256 sidecar next to the archive.
It does not sign or notarize an artifact. Those platform release gates belong
to the CI job that publishes the ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform as host_platform
import stat
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def _architecture(machine: str) -> str:
    value = machine.lower()
    if value in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value


def platform_name(kind: str, machine: str | None = None) -> str:
    """Return the release platform key used by both desktop updaters."""
    kind = kind.lower()
    if kind not in {"macos", "windows"}:
        raise ValueError("platform must be macos or windows")
    return f"{kind}-{_architecture(machine or host_platform.machine())}"


def _iter_source(root: Path) -> Iterable[tuple[Path, str]]:
    """Yield source paths and archive-relative names in stable order."""
    if root.is_symlink():
        raise ValueError("release source must not be a symlink")
    if not root.is_dir():
        raise FileNotFoundError(f"release source directory is missing: {root}")
    yield root, root.name + "/"
    paths = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    for path in paths:
        if path.name in {".DS_Store", "__pycache__"} or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        archive_name = (Path(root.name) / relative).as_posix()
        if path.is_symlink():
            raise ValueError(f"release source contains a symlink: {path}")
        if path.is_dir():
            yield path, archive_name.rstrip("/") + "/"
        elif path.is_file():
            yield path, archive_name
        else:
            raise ValueError(f"release source contains unsupported entry: {path}")


def _zip_info(archive_name: str, *, directory: bool, mode: int) -> ZipInfo:
    info = ZipInfo(archive_name)
    # DOS epoch avoids source filesystem timestamps leaking into the archive.
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.create_system = 3
    file_mode = (stat.S_IFDIR if directory else stat.S_IFREG) | (mode & 0o777)
    info.external_attr = file_mode << 16
    if directory:
        info.external_attr |= 0x10
    info.compress_type = ZIP_DEFLATED
    return info


def create_archive(source: Path, destination: Path) -> Path:
    """Write one deterministic archive rooted at ``source.name``."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for path, archive_name in _iter_source(source):
            if path.is_dir():
                info = _zip_info(archive_name, directory=True, mode=0o755)
                bundle.writestr(info, b"")
            else:
                mode = path.stat().st_mode
                info = _zip_info(archive_name, directory=False, mode=mode)
                bundle.writestr(info, path.read_bytes())
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(path: Path, digest: str) -> Path:
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sidecar


def package(source: Path, output_dir: Path, *, platform: str, version: str) -> dict[str, str | int]:
    version = version.removeprefix("v")
    if not version or any(character.isspace() for character in version):
        raise ValueError("version must be a non-empty tag-safe value")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"ZephyrMesh-{platform}-v{version}.zip"
    create_archive(source, archive)
    digest = sha256_file(archive)
    sidecar = write_checksum(archive, digest)
    return {
        "platform": platform,
        "version": version,
        "filename": archive.name,
        "path": str(archive),
        "sha256": digest,
        "size_bytes": archive.stat().st_size,
        "checksum_path": str(sidecar),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="built application directory")
    parser.add_argument("--output", type=Path, required=True, help="directory for ZIP and SHA-256 sidecar")
    parser.add_argument("--platform", required=True, help="macos, windows, or an explicit platform key")
    parser.add_argument("--version", required=True, help="release version, with or without a leading v")
    args = parser.parse_args(argv)
    platform_key = args.platform
    if "-" not in platform_key:
        platform_key = platform_name(platform_key)
    result = package(args.source, args.output, platform=platform_key, version=args.version)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
