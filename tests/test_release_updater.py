"""Focused checks for the GitHub release updater and safe staging boundary."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import stat
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
import json

import pytest

from tools.release_updater import (
    GitHubReleaseClient,
    ReleaseUpdateError,
    UpdateError,
    apply_staged_update,
    load_manifest,
    platform_key,
    stage_asset,
)


class _Response:
    """Stands in for a urllib response: chunked reads, optional headers, close()."""

    def __init__(self, body: bytes, headers: dict | None = None):
        self.body = body
        self.offset = 0
        self.headers = headers or {}
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        end = len(self.body) if size is None or size < 0 else self.offset + size
        block = self.body[self.offset:end]
        self.offset += len(block)
        return block

    def close(self) -> None:
        self.closed = True


def _release_fixture(*, digest: str | None = None, archive: bytes = b"") -> tuple[dict, bytes]:
    if not archive:
        payload = BytesIO()
        with ZipFile(payload, "w", ZIP_DEFLATED) as bundle:
            bundle.writestr("ZephyrMesh.app/Contents/MacOS/ZephyrMeshApp", b"synthetic app")
        archive = payload.getvalue()
    digest = digest or sha256(archive).hexdigest()
    release = {
        "tag_name": "v0.2.0",
        "name": "Zephyr Mesh 0.2.0",
        "html_url": "https://github.com/Whyater/Zephyr-Mesh/releases/tag/v0.2.0",
        "assets": [{
            "name": "ZephyrMesh-windows-x86_64-v0.2.0.zip",
            "browser_download_url": "https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.0/ZephyrMesh-windows-x86_64-v0.2.0.zip",
            "digest": "sha256:" + digest,
        }],
    }
    return release, archive


def test_release_client_selects_platform_asset_and_reports_update():
    release, archive = _release_fixture()

    def opener(request, timeout):
        if request.full_url.endswith("/releases/latest"):
            return _Response(json.dumps(release).encode())
        return _Response(archive)

    client = GitHubReleaseClient(opener=opener)
    info, asset, newer = client.check("v0.1.0", platform="windows-x86_64")
    assert info.tag == "v0.2.0"
    assert asset.name.startswith("ZephyrMesh-windows")
    assert newer is True


def test_release_client_prefers_runtime_architecture_asset():
    release, _archive = _release_fixture()
    release["assets"].append({
        "name": "ZephyrMesh-windows-arm64-v0.2.0.zip",
        "browser_download_url": "https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.0/ZephyrMesh-windows-arm64-v0.2.0.zip",
        "digest": "sha256:" + "1" * 64,
    })
    from tools.release_updater import ReleaseInfo
    selected = GitHubReleaseClient().select_asset(ReleaseInfo.from_json(release), platform="windows-arm64")
    assert selected.name == "ZephyrMesh-windows-arm64-v0.2.0.zip"


def test_release_client_rejects_mismatched_architecture_asset():
    release, _archive = _release_fixture()
    from tools.release_updater import ReleaseInfo
    parsed = ReleaseInfo.from_json(release)
    with pytest.raises(UpdateError, match="exact windows-arm64"):
        GitHubReleaseClient().select_asset(parsed, platform="windows-arm64")


def test_download_and_stage_verifies_digest_and_safe_payload(tmp_path):
    release, archive = _release_fixture()

    def opener(request, timeout):
        return _Response(archive)

    client = GitHubReleaseClient(opener=opener)
    # Construct through the public parser so this test checks the API contract.
    from tools.release_updater import ReleaseInfo
    parsed = ReleaseInfo.from_json(release)
    staged = client.download_and_stage(parsed.assets[0], staging_parent=tmp_path)
    assert staged.payload.is_dir()
    assert (staged.payload / "Contents" / "MacOS" / "ZephyrMeshApp").read_bytes() == b"synthetic app"
    assert staged.sha256 == sha256(archive).hexdigest()


def test_digest_mismatch_is_rejected_and_staging_is_removed(tmp_path):
    release, archive = _release_fixture(digest="0" * 64)
    from tools.release_updater import ReleaseInfo
    asset = ReleaseInfo.from_json(release).assets[0]

    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(archive))
    with pytest.raises(UpdateError, match="digest mismatch"):
        client.download_and_stage(asset, staging_parent=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_zip_path_traversal_is_rejected(tmp_path):
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as bundle:
        bundle.writestr("ZephyrMesh.app/Contents/MacOS/ZephyrMeshApp", b"safe")
        bundle.writestr("../escaped.txt", b"unsafe")
    release, archive = _release_fixture(archive=payload.getvalue())
    from tools.release_updater import ReleaseInfo
    asset = ReleaseInfo.from_json(release).assets[0]
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(archive))
    with pytest.raises(UpdateError, match="path traversal"):
        client.download_and_stage(asset, staging_parent=tmp_path)


def test_zip_symlink_is_rejected(tmp_path):
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as bundle:
        bundle.writestr("ZephyrMesh.app/Contents/MacOS/ZephyrMeshApp", b"safe")
        link = ZipInfo("ZephyrMesh.app/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(link, b"/tmp/unsafe")
    release, archive = _release_fixture(archive=payload.getvalue())
    from tools.release_updater import ReleaseInfo
    asset = ReleaseInfo.from_json(release).assets[0]
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(archive))
    with pytest.raises(UpdateError, match="symlink"):
        client.download_and_stage(asset, staging_parent=tmp_path)


def test_apply_staged_update_keeps_rollback_backup(tmp_path):
    staged = tmp_path / "staged" / "ZephyrMesh.app"
    staged.mkdir(parents=True)
    (staged / "version.txt").write_text("new", encoding="utf-8")
    install = tmp_path / "installed" / "ZephyrMesh.app"
    install.mkdir(parents=True)
    (install / "version.txt").write_text("old", encoding="utf-8")
    backup = apply_staged_update(staged, install, backup_dir=tmp_path / "rollback-cache")
    assert (install / "version.txt").read_text(encoding="utf-8") == "new"
    assert (backup / "version.txt").read_text(encoding="utf-8") == "old"


def test_offline_manifest_wrapper_verifies_and_stages(tmp_path):
    content = b"offline release asset"
    source = tmp_path / "preview.zip"
    source.write_bytes(content)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "zephyr-release-manifest-1",
        "version": "0.2.0",
        "channel": "stable",
        "assets": [{
            "platform": "windows-x86_64",
            "filename": source.name,
            "sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
            "url": "file://" + str(source),
        }],
    }), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    staged = stage_asset(source, manifest, platform_name="windows-x86_64", staging_dir=tmp_path / "staging")
    assert staged.path.read_bytes() == content
    assert json.loads(staged.metadata_path.read_text(encoding="utf-8"))["verified"] is True
    assert platform_key("Windows", "AMD64") == "windows-x86_64"


def test_offline_manifest_rejects_digest_and_filename_errors(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "zephyr-release-manifest-1",
        "version": "0.2.0",
        "channel": "stable",
        "assets": [{"platform": "windows-x86_64", "filename": "../escape.zip", "sha256": "0" * 64, "size_bytes": 0, "url": "file:///tmp/escape.zip"}],
    }), encoding="utf-8")
    with pytest.raises(ReleaseUpdateError, match="plain file name"):
        load_manifest(manifest_path)


def test_offline_manifest_refuses_nonempty_staging_directory(tmp_path):
    content = b"offline release asset"
    source = tmp_path / "preview.zip"
    source.write_bytes(content)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "zephyr-release-manifest-1",
        "version": "0.2.0",
        "channel": "stable",
        "assets": [{
            "platform": "windows-x86_64",
            "filename": source.name,
            "sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
            "url": "file://" + str(source),
        }],
    }), encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "keep.txt").write_text("do not delete", encoding="utf-8")
    manifest = load_manifest(manifest_path)
    with pytest.raises(ReleaseUpdateError, match="new or empty"):
        stage_asset(source, manifest, platform_name="windows-x86_64", staging_dir=staging)
    assert (staging / "keep.txt").read_text(encoding="utf-8") == "do not delete"


# ---------------------------------------------------------------------------
# Shared cross-language cases. ReleaseUpdaterTests.swift reads the same file, so
# the Python and Swift updaters cannot quietly disagree again.
# ---------------------------------------------------------------------------
CASES = json.loads((Path(__file__).parents[1] / "tools/macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/updater-cases.json").read_text(encoding="utf-8"))


def _sign(a: str, b: str) -> str:
    from tools.release_updater import version_key
    left, right = version_key(a), version_key(b)
    return "greater" if left > right else "less" if left < right else "equal"


@pytest.mark.parametrize("case", CASES["ordering"]["compare"], ids=lambda case: f"{case['a']}-vs-{case['b']}")
def test_version_ordering_matches_shared_cases(case):
    assert _sign(case["a"], case["b"]) == case["expect"]


def test_local_dev_build_is_older_than_matching_release():
    from tools.release_updater import version_key

    assert version_key("0.2.3") > version_key("0.2.3-dev")


@pytest.mark.parametrize("value", CASES["ordering"]["invalid"], ids=repr)
def test_unparseable_versions_are_rejected_not_guessed(value):
    from tools.release_updater import version_key
    with pytest.raises(UpdateError):
        version_key(value)


def test_check_reports_pre_release_current_version_as_older(monkeypatch):
    release, archive = _release_fixture()
    release["tag_name"] = "v0.3.0"
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(json.dumps(release).encode()))
    release["assets"][0]["name"] = "ZephyrMesh-windows-x86_64-v0.3.0.zip"
    _, _, newer = client.check("0.3.0-rc1", platform="windows-x86_64")
    assert newer is True
    _, _, newer = client.check("0.3.0", platform="windows-x86_64")
    assert newer is False


@pytest.mark.parametrize("case", CASES["select"], ids=lambda case: case["name"])
def test_asset_selection_matches_shared_cases(case):
    from tools.release_updater import ReleaseInfo
    base = "https://github.com/Whyater/Zephyr-Mesh/releases/download/" + case["tag"] + "/"
    release = ReleaseInfo.from_json({
        "tag_name": case["tag"], "name": case["tag"], "html_url": "https://github.com/Whyater/Zephyr-Mesh/releases",
        "assets": [{"name": name, "browser_download_url": base + name, "digest": "sha256:" + "a" * 64} for name in case["assets"]],
    })
    client = GitHubReleaseClient()
    if case["expect"] is None:
        with pytest.raises(UpdateError):
            client.select_asset(release, platform=case["platform"])
    else:
        assert client.select_asset(release, platform=case["platform"]).name == case["expect"]


@pytest.mark.parametrize("url", CASES["download_urls"]["allowed"])
def test_official_release_download_urls_are_allowed(url):
    from tools.release_updater import is_official_download_url
    assert is_official_download_url(url, CASES["download_urls"]["repository"])


@pytest.mark.parametrize("url", CASES["download_urls"]["rejected"], ids=repr)
def test_other_hosts_repos_and_odd_urls_are_rejected(url):
    from tools.release_updater import is_official_download_url
    assert not is_official_download_url(url, CASES["download_urls"]["repository"])


def test_download_refuses_a_foreign_url_before_any_request(tmp_path):
    from tools.release_updater import ReleaseAsset
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return _Response(b"never fetched")

    asset = ReleaseAsset(name="ZephyrMesh-windows-x86_64-v0.2.0.zip", url="https://example.invalid/x.zip", digest="sha256:" + "a" * 64)
    with pytest.raises(UpdateError, match="official release"):
        GitHubReleaseClient(opener=opener).download_and_stage(asset, staging_parent=tmp_path)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


# ---- streaming download, size limits, and staging hygiene ------------------


def _asset_for(archive: bytes):
    from tools.release_updater import ReleaseInfo
    release, _ = _release_fixture(archive=archive)
    return ReleaseInfo.from_json(release).assets[0]


def test_download_streams_in_chunks_writes_a_record_and_drops_the_archive(tmp_path):
    from tools.release_updater import STAGED_RECORD, payload_digest, verify_staged_payload
    release, archive = _release_fixture()
    responses = []

    def opener(request, timeout):
        responses.append(_Response(archive))
        return responses[-1]

    asset = _asset_for(archive)
    staged = GitHubReleaseClient(opener=opener).download_and_stage(asset, staging_parent=tmp_path)
    assert responses[0].closed, "the HTTP response is closed"
    record = json.loads((staged.root / STAGED_RECORD).read_text(encoding="utf-8"))
    assert record["tree_sha256"] == payload_digest(staged.payload)
    assert record["archive_sha256"] == sha256(archive).hexdigest()
    assert not (staged.root / asset.name).exists(), "the verified archive is not kept twice"
    verify_staged_payload(staged.payload)


def test_download_refuses_a_declared_size_over_the_limit_before_writing(tmp_path):
    asset = _asset_for(b"")
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(b"x", headers={"Content-Length": str(10**12)}))
    with pytest.raises(UpdateError, match="larger than"):
        client.download_and_stage(asset, staging_parent=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_stops_when_the_body_passes_the_limit(tmp_path, monkeypatch):
    import tools.release_updater as updater
    monkeypatch.setattr(updater, "MAX_ARCHIVE_BYTES", 1024)
    body = b"z" * 4096
    asset = _asset_for(body)
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(body))
    with pytest.raises(UpdateError, match="larger than"):
        client._download(asset.url, tmp_path / "x.zip", limit=updater.MAX_ARCHIVE_BYTES)
    (tmp_path / "staging").mkdir()
    with pytest.raises(UpdateError, match="larger than"):
        client.download_and_stage(asset, staging_parent=tmp_path / "staging")
    assert list((tmp_path / "staging").iterdir()) == [], "the partial download is removed"


def test_release_json_is_size_limited_too():
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(b"{" + b" " * (6 * 1024 * 1024) + b"}"))
    with pytest.raises(UpdateError, match="larger than"):
        client.latest_release()


@pytest.mark.parametrize("limit_name, value, pattern", [("MAX_ARCHIVE_MEMBERS", 2, "entries"), ("MAX_EXTRACTED_BYTES", 100, "expands")])
def test_extraction_is_bounded(tmp_path, monkeypatch, limit_name, value, pattern):
    import tools.release_updater as updater
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as bundle:
        for index in range(3):
            bundle.writestr(f"ZephyrMesh.app/file{index}", b"0" * 1000)  # compresses to almost nothing
    monkeypatch.setattr(updater, limit_name, value)
    asset = _asset_for(payload.getvalue())
    client = GitHubReleaseClient(opener=lambda request, timeout: _Response(payload.getvalue()))
    with pytest.raises(UpdateError, match=pattern):
        client.download_and_stage(asset, staging_parent=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_update_errors_are_short_and_plain():
    from email.message import Message
    from urllib.error import HTTPError, URLError
    import socket
    from tools.release_updater import describe_update_error

    forbidden = HTTPError("https://api.github.com/x", 403, "rate limit exceeded " * 40, Message(), BytesIO(b"x" * 664))
    assert describe_update_error(forbidden).startswith("GitHub refused the request (HTTP 403)")
    offline = URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))
    assert describe_update_error(offline) == "Could not reach GitHub. Check the internet connection."
    assert "Errno" not in describe_update_error(offline)
    assert describe_update_error(KeyError("tag_name")) == "unexpected KeyError"
    assert len(describe_update_error(UpdateError("x" * 1000))) <= 160


def test_sweep_removes_only_old_update_leftovers(tmp_path):
    import os
    from tools.release_updater import sweep_stale_update_files
    old_root = tmp_path / "zephyr-update-old"
    old_root.mkdir()
    new_root = tmp_path / "zephyr-update-new"
    new_root.mkdir()
    old_helper = tmp_path / "ZephyrMeshUpdater-99.exe"
    old_helper.write_bytes(b"helper")
    unrelated = tmp_path / "someone-elses-file.txt"
    unrelated.write_text("keep", encoding="utf-8")
    for path in (old_root, old_helper, unrelated):
        os.utime(path, (1, 1))
    sweep_stale_update_files(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["someone-elses-file.txt", "zephyr-update-new"]


def test_swap_works_across_disks_and_keeps_the_backup_outside_the_install_folder(tmp_path, monkeypatch):
    import errno
    import os
    import tools.release_updater as updater
    real_replace = os.replace

    def replace(source, destination):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    staged = tmp_path / "staged" / "ZephyrMeshWindows"
    staged.mkdir(parents=True)
    (staged / "version.txt").write_text("new", encoding="utf-8")
    install = tmp_path / "disk-d" / "ZephyrMeshWindows"
    install.mkdir(parents=True)
    (install / "version.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(updater.os, "replace", replace)
    try:
        backup = apply_staged_update(staged, install, backup_dir=tmp_path / "disk-c" / "backup")
    finally:
        monkeypatch.setattr(updater.os, "replace", real_replace)
    assert (install / "version.txt").read_text(encoding="utf-8") == "new"
    assert (backup / "version.txt").read_text(encoding="utf-8") == "old"
    assert sorted(path.name for path in install.parent.iterdir()) == ["ZephyrMeshWindows"]


def test_self_test_runner_reports_a_build_that_does_not_pass(tmp_path):
    import sys
    from tools.release_updater import run_self_test
    # The Python interpreter does not know --self-test, exits 2, and prints usage: a real failing process.
    with pytest.raises(UpdateError, match="self-test failed"):
        run_self_test(Path(sys.executable))
    with pytest.raises(UpdateError, match="missing"):
        run_self_test(tmp_path / "nothing.exe")
