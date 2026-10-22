# Install and build

Downloads, local builds and running the simulator from source. The short version is in the [README](../README.md#download).

## macOS

The native app requires macOS 14 or newer. Download the architecture that matches your Mac from [Zephyr Mesh v0.2.4](https://github.com/Whyater/Zephyr-Mesh/releases/tag/v0.2.4):

- [Apple Silicon](https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.4/ZephyrMesh-macos-arm64-v0.2.4.zip)
- [Intel](https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.4/ZephyrMesh-macos-x86_64-v0.2.4.zip)

If you downloaded v0.2.2: that Mac build crashes at launch on any Mac other than the one that built it, because it cannot find its bundled replay file. v0.2.4 includes the fix.

Unzip the download and open `Zephyr Mesh.app`. SHA-256 sidecars are attached to the release. The package is ad hoc signed and may require a one-time confirmation in macOS Privacy & Security. To build the local bundle instead, install the Xcode Command Line Tools and run:

```bash
python3 tools/macos/build_app.py
open "Zephyr Mesh.app"
```

Local builds use `VERSION` with a `-dev` suffix. Add `--install` to copy the
bundle into `~/Applications`; release builds pass `--version` explicitly.

The app does not require a server or network connection to show the bundled replay.

## Windows

Download the [Windows x86_64 package](https://github.com/Whyater/Zephyr-Mesh/releases/download/v0.2.4/ZephyrMesh-windows-x86_64-v0.2.4.zip) from [Zephyr Mesh v0.2.4](https://github.com/Whyater/Zephyr-Mesh/releases/tag/v0.2.4). Extract the complete ZIP before opening `ZephyrMeshWindows\ZephyrMeshWindows.exe`; keep the `_internal` folder beside it. The release also provides a separate .zip.sha256 asset for verification.

The Windows download is unsigned, so Windows SmartScreen may show a warning the first time you run it. Release v0.2.4 adds Windows startup diagnostics (`startup.log` and `README-Windows.txt`) to help with startup failures. Startup problems have not been reproduced on a Windows machine or confirmed fixed.

To build locally, install 64-bit Python 3.12 (the version the release builds use, and the only one tested), open PowerShell at the repository root, and run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/desktop/build_windows.ps1
```

The script runs `py`, which picks the newest Python you have installed. If that is not 3.12, add `-Python` and the full path to your 3.12 `python.exe`. The script creates `dist\ZephyrMeshWindows\ZephyrMeshWindows.exe` and `dist\ZephyrMeshUpdater.exe`. Keep the updater beside the application directory. The app's update action only installs a published release asset with a matching platform architecture and GitHub SHA-256 digest. The digest check is an integrity check only, not publisher authentication: it does not prove who published the file. Releases are not notarized or signed with a developer certificate.

## Run from source

Use Python 3.12 or 3.13. The published runs were made with 3.13, and the automated tests run on 3.12. Python 3.14 has ready-made packages for every pin but has not been tested. The pinned packages in `requirements.txt` will not install on older versions, including the Python 3.9 that comes with macOS, or on Python 3.15. On a Mac with [Homebrew](https://brew.sh), `brew install python@3.12` gives you `python3.12`. If you use 3.13 or 3.14 instead, change the first command below to match.

The commands below are for a bash or zsh terminal, like the macOS Terminal. On Windows, install 64-bit Python 3.12 from [python.org](https://www.python.org/downloads/windows/) and run the commands in Git Bash, which comes with [Git for Windows](https://git-scm.com/downloads/win). Replace the first two commands in the block below with `py -3.12 -m venv .venv` and `source .venv/Scripts/activate`. This Windows route has not been tried on a Windows machine yet.

Get the code with `git clone https://github.com/Whyater/Zephyr-Mesh.git`, then `cd Zephyr-Mesh`. The Python simulator and replay tools run offline from that folder, the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
TMP_DIR=$(mktemp -d)
PYTHONPATH=. python tools/generate_s7_fixture.py --output "$TMP_DIR/s7-swarm"
PYTHONPATH=. python tools/analyze_trace.py sim/trace_example.json
PYTHONPATH=. python tools/analyze_vision_trace.py sim/vision_trace_example.json
PYTHONPATH=. python tools/validate_hardware_profile.py sim/hardware_profile_example.json
PYTHONPATH=. python tools/analyze_bench_trace.py sim/bench_trace_example.json
PYTHONPATH=. python tools/build_evidence_report.py espnow sim/trace_example.json --output /tmp/zephyr-espnow-report.json
PYTHONPATH=. python tools/build_investigation_report.py sim/investigation_example.json --output /tmp/zephyr-s6-report.json
PYTHONPATH=. python tools/analyze_s7_run.py runs/s7-swarm/run.json
PYTHONPATH=. python tools/run_scenario.py --config sim/scenario_example.json --output /tmp/zephyr-s7-scenario-run.json --summary
```

A scenario config may add `velocity_mps` to a keep-out sphere for a bounded linear moving constraint; the generated frame snapshots retain each synthetic center for inspection. The canonical artifacts are [`runs/s0-baseline`](../runs/s0-baseline), [`runs/s2-physics`](../runs/s2-physics), and [`runs/s7-swarm`](../runs/s7-swarm). The native and Windows surfaces consume the S7 replay through [`tools/desktop/replay.py`](../desktop/replay.py) and the checked-in macOS fixture projection. See [`ROADMAP.md`](ROADMAP.md) for evidence gates.

To inspect a trace or profile in either desktop app, create a portable report
with `tools/build_evidence_report.py`, then choose **Open evidence report…**
from the File menu. Reports are read-only. Their declared status and
limitations stay attached to the source summary. A digest identifies the
input file without establishing flight performance.

The scenario-run CLI and strict scenario inspectors described above are built into v0.2.4 but have not yet been checked on separate target machines.
