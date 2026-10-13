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
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile


DEFAULT_REPOSITORY = "Whyater/Zephyr-Mesh"
API_ROOT = "https://api.github.com"

# Hard limits so a hostile or broken release cannot fill memory or disk.
# Real release ZIPs are tens of MB.
MAX_RELEASE_JSON_BYTES = 5 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 50_000
_CHUNK = 1024 * 1024
# Name of the record ``download_and_stage`` writes next to the payload, which
# the helper checks again right before it swaps anything.
STAGED_RECORD = "staged.json"
STAGING_PREFIX = "zephyr-update-"
# How long the helper watches the relaunched app before deleting the backup.
STARTUP_GRACE_S = 5.0


class UpdateError(RuntimeError):
    """Raised when a release cannot be verified or staged safely."""


class ProcessCheckError(UpdateError):
    """The helper could not tell whether the old app is still running.

    The old app may still be open, so the helper must not start a second copy.
    """


class RollbackError(UpdateError):
    """The new app failed and the previous one could not be put back."""


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


_VERSION_PATTERN = re.compile(r"v?(?P<numbers>[0-9]{1,9}(?:\.[0-9]{1,9})*)(?:-(?P<pre>[0-9A-Za-z.\-]+))?(?:\+[0-9A-Za-z.\-]+)?")


def version_key(value: str) -> tuple:
    """Turn a release tag such as ``v0.2.1`` into an ordered, comparable key.

    Rules (the Swift updater implements the same ones, and both are tested
    against ``updater-cases.json``):

    * An optional leading ``v``, then dotted numbers. They compare as numbers
      and trailing zeros are ignored, so ``0.2`` equals ``v0.2.0``.
    * A suffix after a hyphen marks a pre-release. It sorts below the same
      release (``0.3.0-rc1`` is older than ``0.3.0``) and above the previous
      release. Pre-releases compare chunk by chunk, digits as numbers, so
      ``rc10`` is newer than ``rc2``.
    * Build metadata after ``+`` is ignored.
    * Anything else raises ``UpdateError`` instead of being guessed at.
    """
    match = _VERSION_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise UpdateError(f"unrecognized version: {value!r}")
    numbers = [int(part) for part in match.group("numbers").split(".")]
    while numbers and numbers[-1] == 0:
        numbers.pop()
    pre = match.group("pre")
    if pre is None:
        return (tuple(numbers), 1, ())
    chunks = re.findall(r"[0-9]+|[^0-9.\-]+", pre)
    if not chunks or any(chunk[0] in "0123456789" and len(chunk) > 9 for chunk in chunks):
        raise UpdateError(f"unrecognized version: {value!r}")
    # (0, number) sorts before (1, text), so a numeric chunk is older than a word.
    return (tuple(numbers), 0, tuple((0, int(chunk)) if chunk[0] in "0123456789" else (1, chunk) for chunk in chunks))


def describe_update_error(exc: BaseException, *, limit: int = 160) -> str:
    """A short, plain reason for an update failure, for people rather than logs.

    Network errors from urllib carry socket text such as ``[Errno 8] nodename
    nor servname provided``, and an HTTP 403 body from GitHub can run to
    hundreds of characters. Neither helps the person who clicked the button.
    """
    from urllib.error import HTTPError, URLError
    import socket

    if isinstance(exc, HTTPError):
        if exc.code in (403, 429):
            return f"GitHub refused the request (HTTP {exc.code}), usually a rate limit. Try again later."
        if exc.code == 404:
            return "GitHub has no published release to check (HTTP 404)."
        return f"GitHub returned HTTP {exc.code}."
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "GitHub did not answer in time. Check the internet connection."
    if isinstance(exc, URLError):
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            return "GitHub did not answer in time. Check the internet connection."
        return "Could not reach GitHub. Check the internet connection."
    if isinstance(exc, (ConnectionError, socket.gaierror)):
        return "Could not reach GitHub. Check the internet connection."
    if isinstance(exc, UpdateError):
        text = " ".join(str(exc).split())
    elif isinstance(exc, OSError):
        text = f"a file could not be read or written ({exc.strerror or type(exc).__name__})"
    else:
        text = f"unexpected {type(exc).__name__}"
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def normalize_digest(value: str | None) -> str | None:
    if value is None:
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise UpdateError("release digest must be a SHA-256 hex string")
    return digest


def is_official_download_url(url: str, repository: str = DEFAULT_REPOSITORY) -> bool:
    """True only for ``https://github.com/<repository>/releases/download/<something>``.

    The asset URL comes from the same API response as its digest, so a digest
    check alone cannot tell a real release from a forged one. This narrows
    where a download may come from: no other host, repository, port, user info,
    or ``..`` path tricks.
    """
    if not isinstance(url, str) or not url or "\\" in url:
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or parts.netloc != "github.com" or port is not None:
        return False
    prefix = f"/{repository}/releases/download/"
    if not parts.path.startswith(prefix):
        return False
    tail = parts.path[len(prefix):]
    lowered = tail.lower()
    if not tail or "%2e" in lowered or "%2f" in lowered or "%5c" in lowered:
        return False
    return all(segment not in {".", ".."} for segment in tail.split("/"))


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

    def _open(self, url: str, accept: str) -> Any:
        request = Request(url, headers={"Accept": accept, "User-Agent": "Zephyr-Mesh-Updater/0.1"})
        try:
            return self.opener(request, timeout=self.timeout_s)
        except Exception as exc:  # urllib errors vary by platform
            raise UpdateError(describe_update_error(exc)) from exc

    @staticmethod
    def _chunks(response: Any, limit: int) -> Iterable[bytes]:
        """Yield the body in chunks, refusing more than ``limit`` bytes in total."""
        declared = getattr(response, "headers", None)
        length = declared.get("Content-Length") if declared is not None else None
        if length is not None and str(length).isdigit() and int(length) > limit:
            raise UpdateError(f"GitHub response is larger than {limit // (1024 * 1024)} MB")
        total = 0
        while True:
            try:
                block = response.read(_CHUNK)
            except Exception as exc:
                raise UpdateError(describe_update_error(exc)) from exc
            if not block:
                return
            total += len(block)
            if total > limit:
                raise UpdateError(f"GitHub response is larger than {limit // (1024 * 1024)} MB")
            yield block

    @staticmethod
    def _close(response: Any) -> None:
        try:
            response.close()
        except Exception:
            pass

    def _get(self, url: str, *, limit: int | None = None) -> bytes:
        response = self._open(url, "application/vnd.github+json")
        try:
            return b"".join(self._chunks(response, MAX_RELEASE_JSON_BYTES if limit is None else limit))
        finally:
            self._close(response)

    def _download(self, url: str, destination: Path, *, limit: int | None = None) -> str:
        """Stream ``url`` into ``destination`` and return its SHA-256 hex digest."""
        limit = MAX_ARCHIVE_BYTES if limit is None else limit
        digest = hashlib.sha256()
        response = self._open(url, "application/octet-stream")
        try:
            with Path(destination).open("wb") as handle:
                for block in self._chunks(response, limit):
                    digest.update(block)
                    handle.write(block)
        finally:
            self._close(response)
        return digest.hexdigest()

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
        """Return the asset named exactly ``ZephyrMesh-{platform}-v{version}.zip``.

        That is the name ``tools/package_release.py`` writes. Matching the whole
        name (not a substring such as ``win`` inside ``darwin``) means a
        release can never hand a Mac build to a Windows machine or the reverse.
        """
        platform = platform or self.platform_name()
        base, _, architecture = platform.partition("-")
        if base not in {"windows", "macos", "linux"}:
            raise ValueError("platform must be windows, macos, or linux with an optional architecture")
        if not architecture:
            platform = platform_key(base, __import__("platform").machine())
        version = release.tag.removeprefix("v")
        wanted = f"ZephyrMesh-{platform}-v{version}.zip"
        for asset in release.assets:
            if asset.name == wanted:
                return asset
        raise UpdateError(f"release {release.tag} has no exact {platform} zip asset ({wanted})")

    def check(self, current_version: str, *, platform: str | None = None) -> tuple[ReleaseInfo, ReleaseAsset, bool]:
        release = self.latest_release()
        asset = self.select_asset(release, platform=platform)
        return release, asset, version_key(release.tag) > version_key(current_version)

    def download_and_stage(self, asset: ReleaseAsset, *, staging_parent: Path | None = None) -> StagedUpdate:
        expected = normalize_digest(asset.digest)
        if expected is None:
            raise UpdateError(f"release asset {asset.name} has no SHA-256 digest")
        if not is_official_download_url(asset.url, self.repository):
            raise UpdateError(f"download URL is not an official release asset of {self.repository}")
        if Path(asset.name).name != asset.name or asset.name in {"", ".", ".."}:
            raise UpdateError("release asset name must be a plain file name")
        sweep_stale_update_files(Path(staging_parent) if staging_parent is not None else None)
        root = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=staging_parent))
        archive = root / asset.name
        try:
            actual = self._download(asset.url, archive)
            if actual != expected:
                raise UpdateError(f"digest mismatch for {asset.name}: expected {expected}, got {actual}")
            extract = root / "payload"
            extract.mkdir()
            self._safe_extract(archive, extract)
            payload = self._payload_root(extract)
            record = {"archive_sha256": actual, "asset": asset.name, "payload": payload.name, "tree_sha256": payload_digest(payload)}
            (root / STAGED_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            # The archive is no longer needed once the payload is verified and recorded.
            archive.unlink()
            return StagedUpdate(root=root, payload=payload, asset_name=asset.name, sha256=actual)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    @staticmethod
    def _safe_extract(archive: Path, destination: Path) -> None:
        try:
            with ZipFile(archive) as bundle:
                base = destination.resolve()
                members = bundle.infolist()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise UpdateError(f"release archive has more than {MAX_ARCHIVE_MEMBERS} entries")
                # zipfile stops each member at its declared size, so the declared total bounds the disk use.
                if sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
                    raise UpdateError(f"release archive expands to more than {MAX_EXTRACTED_BYTES // (1024 ** 3)} GB")
                for member in members:
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


def default_backup_dir(install_dir: Path) -> Path:
    """Return a per-user rollback location outside the installed app."""
    install_name = Path(install_dir).name or "ZephyrMesh"
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return root / "ZephyrMesh" / "update-backup" / install_name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "org.zephyrmesh.cockpit" / (install_name + ".previous")
    return Path.home() / ".cache" / "zephyrmesh" / (install_name + ".previous")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _is_cross_device(exc: OSError) -> bool:
    # EXDEV on POSIX; ERROR_NOT_SAME_DEVICE (17) on Windows, e.g. C: and D:.
    return exc.errno == errno.EXDEV or getattr(exc, "winerror", None) == 17


def _rename(source: Path, destination: Path) -> bool:
    """Rename in one step. False means the paths are on different disks."""
    try:
        os.replace(source, destination)
    except OSError as exc:
        if _is_cross_device(exc):
            return False
        raise
    return True


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy a folder; on failure remove the partial copy so ``source`` is the only copy."""
    try:
        shutil.copytree(source, destination, symlinks=True)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _move_tree(source: Path, destination: Path) -> None:
    """Rename, or across disks copy and then delete the source (best effort)."""
    if not _rename(source, destination):
        _copy_tree(source, destination)
        shutil.rmtree(source, ignore_errors=True)


def apply_staged_update(staged: StagedUpdate | Path, install_dir: Path, *, backup_dir: Path | None = None) -> Path:
    """Replace an application directory after its process has exited.

    The old directory becomes the rollback backup (outside the install folder)
    until the caller has confirmed the new version starts. If moving the new
    app into place fails, even halfway through a copy between disks, the
    partial copy is deleted and the old app is put back before this raises.
    """
    payload = staged.payload if isinstance(staged, StagedUpdate) else Path(staged)
    install_dir = Path(install_dir).resolve()
    payload = payload.resolve()
    if not payload.is_dir() or payload == install_dir:
        raise UpdateError("staged payload must be a separate directory")
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = (backup_dir or default_backup_dir(install_dir)).resolve()
    if backup == install_dir or backup == payload:
        raise UpdateError("rollback backup must be separate from the app")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        _remove_path(backup)
    moved_old = False
    try:
        if install_dir.exists():
            if not _rename(install_dir, backup):
                _copy_tree(install_dir, backup)
                moved_old = True
                _remove_path(install_dir)
            moved_old = True
        _move_tree(payload, install_dir)
    except Exception as exc:
        if moved_old:
            try:
                rollback_update(backup, install_dir)
            except Exception as rollback_exc:
                raise RollbackError(f"{exc}; the previous version could not be put back ({rollback_exc}) and is saved at {backup}") from exc
        raise
    return backup


def rollback_update(backup: Path, install_dir: Path) -> None:
    """Delete whatever is at ``install_dir`` and move the backup back there."""
    backup = Path(backup).resolve()
    install_dir = Path(install_dir).resolve()
    if not backup.is_dir():
        raise UpdateError("rollback backup is missing")
    if install_dir.exists() or install_dir.is_symlink():
        _remove_path(install_dir)
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    _move_tree(backup, install_dir)


def staged_executable(staged: StagedUpdate | Path, install_dir: Path, launch: Path) -> Path | None:
    """Map the installed executable path into an extracted payload."""
    payload = staged.payload if isinstance(staged, StagedUpdate) else Path(staged)
    payload = payload.resolve()
    if not payload.is_dir():
        return None
    try:
        relative = Path(launch).resolve().relative_to(Path(install_dir).resolve())
    except ValueError:
        relative = Path(launch).name
    candidate = payload / relative
    return candidate if candidate.is_file() else None


def run_self_test(executable: Path, *, timeout_s: float = 20.0) -> None:
    """Run a packaged executable's bounded startup check before installing it."""
    executable = Path(executable)
    if not executable.is_file():
        raise UpdateError(f"self-test executable is missing: {executable}")
    try:
        result = subprocess.run(
            [str(executable), "--self-test"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(f"self-test timed out after {timeout_s:g}s") from exc
    except OSError as exc:
        raise UpdateError(f"self-test could not start: {exc}") from exc
    output = " ".join((result.stdout or "").split() + (result.stderr or "").split()).strip()
    if result.returncode != 0:
        detail = output[:400] or f"exit code {result.returncode}"
        raise UpdateError(f"self-test failed: {detail}")


# ---- the detached helper's steps ------------------------------------------


def payload_digest(payload: Path) -> str:
    """One SHA-256 over every file name and file digest in a folder, in a fixed order."""
    base = Path(payload)
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        if path.is_symlink():
            raise UpdateError("staged update contains a symlink")
        if path.is_dir():
            digest.update(b"d\0" + relative + b"\0")
        else:
            digest.update(b"f\0" + relative + b"\0" + sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def verify_staged_payload(staged: Path) -> None:
    """Check, right before the swap, that ``staged`` is exactly what was verified.

    Only the layout ``download_and_stage`` writes is accepted:
    ``<temp>/zephyr-update-*/payload/<app folder>`` with ``staged.json`` in the
    ``zephyr-update-*`` folder. The folder digest catches files that changed or
    went missing after download (for example a file removed by antivirus). It
    is not protection against malware running as the same user, which could
    rewrite the record too, or simply edit the installed app.
    """
    staged = Path(staged)
    root = staged.parent.parent
    if staged.parent.name != "payload" or not root.name.startswith(STAGING_PREFIX) or not staged.is_dir():
        raise UpdateError("staged update is not in a verified staging folder")
    try:
        record = json.loads((root / STAGED_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UpdateError("staged update has no verification record") from exc
    if not isinstance(record, dict) or record.get("payload") != staged.name or not isinstance(record.get("tree_sha256"), str):
        raise UpdateError("staged update verification record does not match")
    if payload_digest(staged) != record["tree_sha256"]:
        raise UpdateError("staged update changed after it was verified")


def remove_staging_root(staged: Path) -> None:
    """Delete the ``zephyr-update-*`` folder that holds ``staged``. Never anything else."""
    root = Path(staged).parent.parent
    if Path(staged).parent.name == "payload" and root.name.startswith(STAGING_PREFIX) and (root / STAGED_RECORD).is_file():
        shutil.rmtree(root, ignore_errors=True)


def sweep_stale_update_files(directory: Path | None = None, *, max_age_s: float = 3600.0) -> None:
    """Remove staging folders and helper copies left by earlier updates.

    A helper that crashed, an update refused after download, or the helper
    copy that cannot delete itself while it runs all leave files in the temp
    folder. Anything older than ``max_age_s`` is no longer in use.
    """
    folder = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    cutoff = time.time() - max_age_s
    try:
        entries = list(folder.iterdir())
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if not (name.startswith(STAGING_PREFIX) or (name.startswith("ZephyrMeshUpdater-") and name.endswith(".exe"))):
            continue
        try:
            if entry.is_symlink() or entry.lstat().st_mtime > cutoff:
                continue
            _remove_path(entry)
        except OSError:
            continue


def pid_listed_in_tasklist(output: str, pid: int) -> bool:
    """Parse ``tasklist /FO CSV /NH`` so PID 12 cannot match PID 112."""
    for row in csv.reader(output.splitlines()):
        if len(row) >= 2 and row[1].strip() == str(pid):
            return True
    return False


def process_running(pid: int, *, timeout_s: float = 10.0, windows: bool | None = None) -> bool:
    """True while ``pid`` exists. Raises ProcessCheckError when that cannot be told.

    ``windows`` defaults to the running OS; tests set it to exercise the
    ``tasklist`` path anywhere.
    """
    if pid <= 0:
        return False
    if os.name == "nt" if windows is None else windows:
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProcessCheckError(f"tasklist did not answer within {timeout_s:g}s") from exc
        except OSError as exc:
            raise ProcessCheckError(f"could not run tasklist: {exc}") from exc
        if result.returncode != 0:
            raise ProcessCheckError(f"tasklist failed with exit code {result.returncode}")
        return pid_listed_in_tasklist(result.stdout, pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OverflowError:
        return False
    return True


def wait_for_exit(pid: int, timeout_s: float = 45.0) -> None:
    """Wait for ``pid`` to exit; TimeoutError if it is still running at the deadline."""
    deadline = time.monotonic() + timeout_s
    while process_running(pid, timeout_s=min(10.0, max(1.0, timeout_s))):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for process {pid}")
        time.sleep(0.25)


def launch_and_confirm(launch: Path, *, grace_s: float | None = None) -> None:
    """Start the new app and watch it for ``grace_s`` (default STARTUP_GRACE_S).

    Raises if it cannot be started or exits with an error in that time.
    """
    grace_s = STARTUP_GRACE_S if grace_s is None else grace_s
    try:
        process = subprocess.Popen([str(launch)], close_fds=True)
    except OSError as exc:
        raise UpdateError(f"the updated app could not be started: {exc}") from exc
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            if code != 0:
                raise UpdateError(f"the updated app quit right after it started (exit code {code})")
            return  # a clean exit means it started and the user closed it
        time.sleep(0.25)


def install_staged_update(staged: Path, install_dir: Path, launch: Path, *, backup_dir: Path | None = None,
                          grace_s: float | None = None) -> None:
    """Everything the detached helper does once the old app has exited.

    Order: re-check the staged files, run the new build's ``--self-test``,
    swap (old app to a backup outside the install folder), start the new app
    and watch it briefly, then delete the backup. Any failure before the swap
    leaves the installed app untouched; any failure after it puts the old app
    back. Raises on failure; the caller records the reason.
    """
    verify_staged_payload(staged)
    candidate = staged_executable(staged, install_dir, launch)
    if candidate is None:
        raise UpdateError("the staged update is missing its executable")
    run_self_test(candidate)
    backup = apply_staged_update(staged, install_dir, backup_dir=backup_dir)
    try:
        launch_and_confirm(launch, grace_s=grace_s)
    except Exception as exc:
        try:
            rollback_update(backup, install_dir)
        except Exception as rollback_exc:
            raise RollbackError(f"{exc}; the previous version could not be put back ({rollback_exc}) and is saved at {backup}") from exc
        raise
    try:
        if backup.exists():
            _remove_path(backup)
    except OSError:
        pass  # a leftover backup in the cache folder is harmless


def run_update_helper(pid: int, staged: Path, install_dir: Path, launch: Path, *, wait: Callable[[int], None] | None = None,
                      backup_dir: Path | None = None) -> int:
    """Shared body of the detached helpers: wait, install, record the outcome.

    Returns 0 on success, 1 on failure. On failure the result file says why
    and the old app is started again, unless it may still be running (timeout
    or an unanswerable process check) or could not be put back.
    """
    try:
        (wait or wait_for_exit)(pid)
        install_staged_update(staged, install_dir, launch, backup_dir=backup_dir)
    except Exception as exc:
        if isinstance(exc, RollbackError):
            write_update_result(f"The update failed and the previous version could not be put back. {type(exc).__name__}: {exc}")
        else:
            write_update_result(f"The update could not start, so the current version was kept. {type(exc).__name__}: {exc}")
        if not isinstance(exc, (TimeoutError, ProcessCheckError, RollbackError)):
            try:
                subprocess.Popen([str(launch)], close_fds=True)
            except OSError:
                pass
        return 1
    finally:
        remove_staging_root(staged)
    clear_update_result()
    return 0


def update_result_path() -> Path:
    """Per-user file where the detached update helper records a failed update.

    The helper has no console and the app that started it has already exited,
    so this file is the only place a failure can be written down. It lives next
    to ``startup.log`` and outside the folder the updater swaps.
    """
    app_data = os.environ.get("LOCALAPPDATA")
    base = Path(app_data) / "ZephyrMesh" if app_data else Path.home() / ".zephyr-mesh"
    return base / "update-result.txt"


def write_update_result(message: str) -> None:
    """Best effort: a logging failure must not hide the original update error."""
    path = update_result_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message + "\n", encoding="utf-8")
    except OSError:
        pass


def clear_update_result() -> None:
    try:
        update_result_path().unlink(missing_ok=True)
    except OSError:
        pass


def take_update_result() -> str | None:
    """Return the recorded failure once and delete it, or None if there is none."""
    path = update_result_path()
    try:
        message = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    clear_update_result()
    return message or None


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
