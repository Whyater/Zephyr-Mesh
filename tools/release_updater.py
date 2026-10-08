"""Release based updater shared by the Windows preview and macOS bundle.

The updater deliberately consumes GitHub *release assets*, not an arbitrary
branch checkout. A release must publish a platform asset with a SHA-256 digest
(``digest`` in the GitHub API response). Downloads
are staged and verified before an install replaces anything. The running app
must exit before the staged directory is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile


DEFAULT_REPOSITORY = "Whyater/Zephyr-Mesh"
API_ROOT = "https://api.github.com"


class UpdateError(RuntimeError):
    """Raised when a release cannot be verified or staged safely."""


# Backwards-compatible name for local manifest tooling. The GitHub API and
# release-asset path below remain the preferred application update flow.
ReleaseUpdateError = UpdateError


@dataclass(frozen=True)
class ManifestAsset:
    platform: str
    filename: str
    sha256: str
    size_bytes: int
    url: str


@dataclass(frozen=True)
class ReleaseManifest:
    schema: str
    version: str
    channel: str
    assets: tuple[ManifestAsset, ...]

    def asset_for(self, platform: str) -> ManifestAsset:
        for asset in self.assets:
            if asset.platform == platform:
                return asset
        raise ReleaseUpdateError(f"no release asset for {platform}")


@dataclass(frozen=True)
class StagedAsset:
    path: Path
    metadata_path: Path


def platform_key(system: str, machine: str) -> str:
    system = str(system).lower()
    machine = str(machine).lower()
    os_name = "windows" if system.startswith("win") else "macos" if system in {"darwin", "mac", "macos"} else "linux" if system.startswith("linux") else system
    arch = "x86_64" if machine in {"amd64", "x86_64", "x64"} else "arm64" if machine in {"arm64", "aarch64"} else machine
    return f"{os_name}-{arch}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> ReleaseManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseUpdateError(f"could not read release manifest: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != "zephyr-release-manifest-1":
        raise ReleaseUpdateError("unsupported release manifest schema")
    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, list) or not assets_raw:
        raise ReleaseUpdateError("release manifest assets must be a non-empty list")
    assets: list[ManifestAsset] = []
    for item in assets_raw:
        if not isinstance(item, dict):
            raise ReleaseUpdateError("release manifest asset must be an object")
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ReleaseUpdateError("release asset filename must be a plain file name")
        platform = item.get("platform")
        sha = normalize_digest(item.get("sha256"))
        size = item.get("size_bytes")
        url = item.get("url")
        if not isinstance(platform, str) or not platform or sha is None or isinstance(size, bool) or not isinstance(size, int) or size < 0 or not isinstance(url, str) or not url:
            raise ReleaseUpdateError("release manifest asset is incomplete")
        assets.append(ManifestAsset(platform, filename, sha, size, url))
    version = raw.get("version")
    channel = raw.get("channel")
    if not isinstance(version, str) or not version or not isinstance(channel, str) or not channel:
        raise ReleaseUpdateError("release manifest identity is incomplete")
    return ReleaseManifest(raw["schema"], version, channel, tuple(assets))


def verify_asset(path: Path, asset: ManifestAsset) -> None:
    path = Path(path)
    if not path.is_file():
        raise ReleaseUpdateError(f"release asset is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != asset.size_bytes:
        raise ReleaseUpdateError(f"size mismatch for {asset.filename}")
    if sha256_file(path) != asset.sha256:
        raise ReleaseUpdateError(f"sha256 mismatch for {asset.filename}")


def stage_asset(path: Path, manifest: ReleaseManifest, *, platform_name: str, staging_dir: Path) -> StagedAsset:
    asset = manifest.asset_for(platform_name)
    verify_asset(path, asset)
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        if not staging_dir.is_dir() or any(staging_dir.iterdir()):
            raise ReleaseUpdateError("staging directory must be new or empty")
    else:
        staging_dir.mkdir(parents=True)
    staged_path = staging_dir / asset.filename
    shutil.copy2(path, staged_path)
    metadata_path = staging_dir / "update.json"
    metadata_path.write_text(json.dumps({"verified": True, "staged_only": True, "version": manifest.version, "platform": platform_name, "sha256": asset.sha256}, indent=2, sort_keys=True), encoding="utf-8")
    return StagedAsset(staged_path, metadata_path)


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    digest: str | None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ReleaseAsset":
        name = raw.get("name")
        url = raw.get("browser_download_url")
        digest = raw.get("digest")
        if not isinstance(name, str) or not name or not isinstance(url, str) or not url:
            raise UpdateError("release asset is missing name or download URL")
        if digest is not None and (not isinstance(digest, str) or not digest):
            raise UpdateError(f"invalid digest for release asset {name}")
        return cls(name=name, url=url, digest=digest)


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    name: str
    html_url: str
    assets: tuple[ReleaseAsset, ...]

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ReleaseInfo":
        tag = raw.get("tag_name")
        name = raw.get("name") or tag
        url = raw.get("html_url")
        assets = raw.get("assets", [])
        if not isinstance(tag, str) or not tag or not isinstance(name, str) or not isinstance(url, str):
            raise UpdateError("latest GitHub release has an invalid identity")
        if not isinstance(assets, list):
            raise UpdateError("latest GitHub release has invalid assets")
        return cls(tag=tag, name=name, html_url=url, assets=tuple(ReleaseAsset.from_json(a) for a in assets))


@dataclass(frozen=True)
class StagedUpdate:
    root: Path
    payload: Path
    asset_name: str
    sha256: str


def version_key(value: str) -> tuple[int, ...]:
    """Turn tags such as ``v0.2.1`` into a comparable numeric tuple."""
    numbers = tuple(int(part) for part in re.findall(r"\d+", value))
    return numbers or (0,)


def normalize_digest(value: str | None) -> str | None:
    if value is None:
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise UpdateError("release digest must be a SHA-256 hex string")
    return digest


class GitHubReleaseClient:
    """Small dependency-free GitHub release client with injectable HTTP."""

    def __init__(self, repository: str = DEFAULT_REPOSITORY, *, opener: Callable[..., Any] = urlopen, timeout_s: float = 15.0) -> None:
        if not repository or "/" not in repository:
            raise ValueError("repository must be OWNER/NAME")
        self.repository = repository
        self.opener = opener
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def _get(self, url: str) -> bytes:
        request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "Zephyr-Mesh-Updater/0.1"})
        try:
            response = self.opener(request, timeout=self.timeout_s)
            return response.read()
        except Exception as exc:  # urllib errors vary by platform
            raise UpdateError(f"GitHub request failed: {exc}") from exc

    def latest_release(self) -> ReleaseInfo:
        url = f"{API_ROOT}/repos/{self.repository}/releases/latest"
        try:
            raw = json.loads(self._get(url).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpdateError("GitHub returned invalid release JSON") from exc
        if not isinstance(raw, dict):
            raise UpdateError("GitHub release response is not an object")
        return ReleaseInfo.from_json(raw)

    @staticmethod
    def platform_name() -> str:
        import sys
        return platform_key(sys.platform, __import__("platform").machine())

    def select_asset(self, release: ReleaseInfo, *, platform: str | None = None) -> ReleaseAsset:
        platform = platform or self.platform_name()
        base, _, architecture = platform.partition("-")
        tokens = {
            "windows": ("windows", "win"),
            "macos": ("macos", "darwin", "mac"),
            "linux": ("linux",),
        }.get(base)
        if tokens is None:
            raise ValueError("platform must be windows, macos, or linux with an optional architecture")
        candidates = [asset for asset in release.assets if asset.name.lower().endswith(".zip") and any(token in asset.name.lower() for token in tokens)]
        if architecture:
            arch_tokens = {"x86_64": ("x86_64", "amd64", "x64"), "arm64": ("arm64", "aarch64")}.get(architecture, (architecture,))
            preferred = [asset for asset in candidates if any(token in asset.name.lower() for token in arch_tokens)]
            universal = [asset for asset in candidates if any(token in asset.name.lower() for token in ("universal", "any-arch", "any_arch"))]
            if preferred:
                candidates = preferred
            elif universal:
                candidates = universal
            else:
                raise UpdateError(f"release {release.tag} has no exact {platform} zip asset")
        if not candidates:
            raise UpdateError(f"release {release.tag} has no {platform} zip asset")
        return sorted(candidates, key=lambda asset: asset.name)[0]

    def check(self, current_version: str, *, platform: str | None = None) -> tuple[ReleaseInfo, ReleaseAsset, bool]:
        release = self.latest_release()
        asset = self.select_asset(release, platform=platform)
        return release, asset, version_key(release.tag) > version_key(current_version)

    def download_and_stage(self, asset: ReleaseAsset, *, staging_parent: Path | None = None) -> StagedUpdate:
        expected = normalize_digest(asset.digest)
        if expected is None:
            raise UpdateError(f"release asset {asset.name} has no SHA-256 digest")
        root = Path(tempfile.mkdtemp(prefix="zephyr-update-", dir=staging_parent))
        archive = root / asset.name
        try:
            archive.write_bytes(self._get(asset.url))
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            if actual != expected:
                raise UpdateError(f"digest mismatch for {asset.name}: expected {expected}, got {actual}")
            extract = root / "payload"
            extract.mkdir()
            self._safe_extract(archive, extract)
            payload = self._payload_root(extract)
            return StagedUpdate(root=root, payload=payload, asset_name=asset.name, sha256=actual)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    @staticmethod
    def _safe_extract(archive: Path, destination: Path) -> None:
        try:
            with ZipFile(archive) as bundle:
                base = destination.resolve()
                for member in bundle.infolist():
                    mode = (member.external_attr >> 16) & 0o170000
                    if stat.S_ISLNK(mode):
                        raise UpdateError("release archive contains a symlink entry")
                    target = (destination / member.filename).resolve()
                    if os.path.commonpath((str(base), str(target))) != str(base):
                        raise UpdateError("release archive contains a path traversal entry")
                bundle.extractall(destination)
        except BadZipFile as exc:
            raise UpdateError("release asset is not a valid zip archive") from exc

    @staticmethod
    def _payload_root(extract: Path) -> Path:
        entries = [entry for entry in extract.iterdir() if entry.name != "__MACOSX"]
        if len(entries) != 1 or not entries[0].is_dir():
            raise UpdateError("release archive must contain exactly one application directory")
        return entries[0]


def apply_staged_update(staged: StagedUpdate | Path, install_dir: Path, *, backup_dir: Path | None = None) -> Path:
    """Replace an application directory after its process has exited.

    The old directory is retained as a rollback backup until the caller has
    successfully relaunched the new version.
    """
    payload = staged.payload if isinstance(staged, StagedUpdate) else Path(staged)
    install_dir = Path(install_dir).resolve()
    payload = payload.resolve()
    if not payload.is_dir() or payload == install_dir:
        raise UpdateError("staged payload must be a separate directory")
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = (backup_dir or install_dir.with_name(install_dir.name + ".previous")).resolve()
    if backup.exists():
        shutil.rmtree(backup)
    if install_dir.exists():
        os.replace(install_dir, backup)
    try:
        os.replace(payload, install_dir)
    except Exception:
        if not install_dir.exists() and backup.exists():
            os.replace(backup, install_dir)
        raise
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or stage a verified Zephyr Mesh GitHub release")
    parser.add_argument("command", choices=("check", "stage"))
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--platform", choices=("windows", "macos", "linux"))
    parser.add_argument("--staging-parent", type=Path)
    args = parser.parse_args()
    client = GitHubReleaseClient(args.repository)
    release, asset, newer = client.check(args.version, platform=args.platform)
    result: dict[str, Any] = {"tag": release.tag, "asset": asset.name, "newer": newer, "release_url": release.html_url}
    if args.command == "stage" and newer:
        staged = client.download_and_stage(asset, staging_parent=args.staging_parent)
        result.update({"staged_root": str(staged.root), "payload": str(staged.payload), "sha256": staged.sha256})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
