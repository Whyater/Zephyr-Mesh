param(
    [string]$Python = "py",
    [string]$Output = "dist"
)

# Run this on Windows from the repository root. PyInstaller produces a
# self-contained GUI executable and a small detached updater beside it.
& $Python -m pip install pyinstaller
& $Python -m PyInstaller --noconfirm --clean --windowed --name ZephyrMeshWindows `
    --paths "." --hidden-import tools.release_updater --add-data "runs\s7-swarm\run.json;runs\s7-swarm" --distpath "$Output" desktop\windows_preview.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m PyInstaller --noconfirm --clean --console --onefile --name ZephyrMeshUpdater `
    --paths "." --distpath "$Output" --workpath "$Output\build-updater" desktop\windows_updater.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built $Output\ZephyrMeshWindows\ZephyrMeshWindows.exe"
Write-Host "Built $Output\ZephyrMeshUpdater.exe (keep beside ZephyrMeshWindows directory)"
Write-Host "The app is replay-only until a digest-verified GitHub release asset is published."
