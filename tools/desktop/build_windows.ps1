param(
    [string]$Python = "py",
    [string]$Output = "dist",
    [string]$Version = "",
    [switch]$Package
)

# Run this on Windows from the repository root. PyInstaller produces a
# self-contained GUI executable and a small detached updater beside it.
$repoRoot = (Get-Location).Path
if ([string]::IsNullOrWhiteSpace($Version)) {
    # Local builds are <VERSION>-dev, the same rule as tools/macos/build_app.py.
    # VERSION is the only source; without it the build stops instead of guessing.
    $versionFile = Join-Path $repoRoot "VERSION"
    if (-not (Test-Path $versionFile)) {
        Write-Error "VERSION is missing. Run this script from the repository root, or pass -Version."
        exit 1
    }
    $Version = (Get-Content -Raw $versionFile).Trim() + "-dev"
}
& $Python -m pip install --requirement tools\desktop\requirements-windows.txt
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$iconArgs = @()
$iconPath = Join-Path $repoRoot "tools\design\icon\ZephyrMesh.ico"
if (Test-Path $iconPath) {
    $iconArgs = @("--icon", $iconPath)
} else {
    Write-Warning "Windows icon is missing; continuing without --icon"
}
$pngArgs = @()
$pngPath = Join-Path $repoRoot "tools\design\icon\ZephyrMesh-256.png"
if (Test-Path $pngPath) {
    $pngArgs = @("--add-data", "$pngPath;tools\design\icon")
} else {
    Write-Warning "Windows icon PNG is missing; continuing without bundled PNG"
}
& $Python -m PyInstaller --noconfirm --clean --windowed --name ZephyrMeshWindows `
    --paths "." @iconArgs @pngArgs `
    --hidden-import tools.release_updater --hidden-import tools.desktop.replay --hidden-import tools.desktop.scenario_run `
    --hidden-import tools.desktop.design_tokens --hidden-import tools.desktop.evidence --hidden-import tools.desktop.scene_projection `
    --hidden-import tools.desktop.ui_settings --hidden-import tools.desktop.replay_session --hidden-import tools.desktop.presentation `
    --hidden-import tools.desktop.ui_format --hidden-import tools.desktop.ui_icons --hidden-import tools.desktop.ui_theme `
    --hidden-import tools.desktop.ui_widgets --hidden-import tools.desktop.scene_view --hidden-import tools.desktop.ui_viewers `
    --hidden-import tools.desktop.results_data --hidden-import tools.desktop.ui_results_chart `
    --hidden-import tools.desktop.ui_results_atlas `
    --hidden-import tools.desktop.ui_results `
    --add-data "sim\runs\s7-swarm\run.json;sim\runs\s7-swarm" `
    --add-data "tools\desktop\evidence_report_example.json;tools\desktop" `
    --add-data "tools\desktop\investigation_report_example.json;tools\desktop" `
    --add-data "tools\desktop\s7_report_example.json;tools\desktop" `
    --add-data "tools\desktop\results_fixture.json;tools\desktop" `
    --distpath "$Output" tools\desktop\windows_preview.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m PyInstaller --noconfirm --clean --console --onefile --name ZephyrMeshUpdater `
    --paths "." --distpath "$Output" --workpath "$Output\build-updater" tools\desktop\windows_updater.py
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
