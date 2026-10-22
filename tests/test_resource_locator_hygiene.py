from pathlib import Path


def test_swift_sources_do_not_use_asserting_bundle_module_accessor() -> None:
    root = Path(__file__).resolve().parents[1] / "tools" / "macos" / "ZephyrMeshApp" / "Sources"
    offenders = [str(path) for path in root.rglob("*.swift") if "Bundle.module" in path.read_text(encoding="utf-8")]
    assert not offenders, f"Bundle.module must not appear in Swift sources: {offenders}"
