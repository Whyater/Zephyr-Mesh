#!/usr/bin/env python3
"""Create the public manifest that accompanies a Zephyr Mesh release.

The updater currently selects and verifies GitHub release assets directly from
the API. This manifest is a human- and script-readable index for mirrors and
release pages; its checksums are independently recomputed from the ZIP files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "zephyr-release-manifest-1"
PLATFORM_RE = re.compile(r"ZephyrMesh-(?P<platform>[a-z0-9_-]+)-v(?P<version>[^/]+)\.zip$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    assets: list[Path],
    *,
    version: str,
    repository: str,
    channel: str = "stable",
) -> dict[str, object]:
    version = version.removeprefix("v")
    if not version:
        raise ValueError("version must be non-empty")
    if not repository or repository.count("/") != 1:
        raise ValueError("repository must be OWNER/NAME")
    if not assets:
        raise ValueError("at least one release ZIP is required")
    manifest_assets: list[dict[str, object]] = []
    seen_platforms: set[str] = set()
    for source in sorted((Path(item).resolve() for item in assets), key=lambda item: item.name):
        if not source.is_file() or source.suffix.lower() != ".zip":
            raise ValueError(f"release asset is not a ZIP file: {source}")
        match = PLATFORM_RE.fullmatch(source.name)
        if match is None:
            raise ValueError(f"release asset name must match ZephyrMesh-<platform>-v<version>.zip: {source.name}")
        platform = match.group("platform")
        if match.group("version") != version:
            raise ValueError(f"release asset version does not match manifest version: {source.name}")
        if platform in seen_platforms:
            raise ValueError(f"duplicate release platform: {platform}")
        seen_platforms.add(platform)
        manifest_assets.append({
            "platform": platform,
            "filename": source.name,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "url": f"https://github.com/{repository}/releases/download/v{version}/{source.name}",
        })
    return {
        "schema": SCHEMA,
        "version": version,
        "channel": channel,
        "assets": manifest_assets,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="release version, with or without a leading v")
    parser.add_argument("--repository", default="Whyater/Zephyr-Mesh")
    parser.add_argument("--channel", default="stable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("assets", nargs="+", type=Path, help="release ZIP assets")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.assets, version=args.version, repository=args.repository, channel=args.channel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output.resolve()), "assets": len(manifest["assets"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
