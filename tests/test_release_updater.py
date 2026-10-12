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
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body


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
    backup = apply_staged_update(staged, install)
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
CASES = json.loads((Path(__file__).parents[1] / "macos/ZephyrMeshApp/Tests/ZephyrMeshAppTests/Fixtures/updater-cases.json").read_text(encoding="utf-8"))


def _sign(a: str, b: str) -> str:
    from tools.release_updater import version_key
    left, right = version_key(a), version_key(b)
    return "greater" if left > right else "less" if left < right else "equal"


@pytest.mark.parametrize("case", CASES["ordering"]["compare"], ids=lambda case: f"{case['a']}-vs-{case['b']}")
def test_version_ordering_matches_shared_cases(case):
    assert _sign(case["a"], case["b"]) == case["expect"]


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
