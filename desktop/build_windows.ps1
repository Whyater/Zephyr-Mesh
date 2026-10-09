param(
    [string]$Python = "py",
    [string]$Output = "dist",
    [string]$Version = "0.1.0",
    [switch]$Package
)

# Run this on Windows from the repository root. PyInstaller produces a
# self-contained GUI executable and a small detached updater beside it.
& $Python -m pip install --requirement requirements-windows.txt
New-Item -ItemType Directory -Force -Path $Output | Out-Null
& $Python -m PyInstaller --noconfirm --clean --windowed --name ZephyrMeshWindows `
    --paths "." --hidden-import tools.release_updater --hidden-import desktop.replay --hidden-import desktop.scenario_run --hidden-import desktop.design_tokens --hidden-import desktop.evidence --add-data "runs\s7-swarm\run.json;runs\s7-swarm" --add-data "desktop\evidence_report_example.json;desktop" --add-data "desktop\investigation_report_example.json;desktop" --add-data "desktop\s7_report_example.json;desktop" --distpath "$Output" desktop\windows_preview.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m PyInstaller --noconfirm --clean --console --onefile --name ZephyrMeshUpdater `
    --paths "." --distpath "$Output" --workpath "$Output\build-updater" desktop\windows_updater.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$versionPath = Join-Path $Output "ZephyrMeshWindows\VERSION.txt"
[System.IO.File]::WriteAllText($versionPath, $Version.TrimStart('v'), [System.Text.UTF8Encoding]::new($false))
$readmePath = Join-Path $Output "ZephyrMeshWindows\README-Windows.txt"
$readme = @(
    "Zephyr Mesh Windows desktop preview"
    ""
    "Extract the complete ZIP before opening ZephyrMeshWindows.exe. Keep the _internal folder beside the executable."
    "If startup fails after opening from File Explorer's ZIP view, extract the complete ZIP first; some setups may omit required support files."
    ""
    "If startup still fails, check %LOCALAPPDATA%\ZephyrMesh\startup.log for a diagnostic report."
    "This build is a read-only synthetic replay. It does not connect to a radio, camera, motor, flight controller, or swarm."
) -join [Environment]::NewLine
[System.IO.File]::WriteAllText($readmePath, $readme + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
Write-Host "Built $Output\ZephyrMeshWindows\ZephyrMeshWindows.exe"
Write-Host "Built $Output\ZephyrMeshUpdater.exe (keep beside ZephyrMeshWindows directory)"
Write-Host "The app is replay-only until a digest-verified GitHub release asset is published."

if ($Package) {
    # A release ZIP contains one application directory so the updater can
    # safely replace it. Include the detached helper inside that directory;
    # the preview also accepts the sibling layout used by local builds.
    Copy-Item "$Output\ZephyrMeshUpdater.exe" "$Output\ZephyrMeshWindows\ZephyrMeshUpdater.exe" -Force
    & $Python tools\package_release.py --source "$Output\ZephyrMeshWindows" --output "$Output\release" --platform windows-x86_64 --version $Version
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Remove-Item "$Output\ZephyrMeshWindows\ZephyrMeshUpdater.exe" -Force
    Write-Host "Packaged $Output\release\ZephyrMesh-windows-x86_64-v$($Version.TrimStart('v')).zip"
}
