"""Small local Zephyr S2 replay server.

The server intentionally exposes only a synthetic, replay-only baseline. It has no
radio, camera, controller, or swarm command path.
"""
from __future__ import annotations
import argparse, csv, io, json, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import numpy as np
from sim.baseline import BaselineConfig, run_baseline, write_baseline
from sim.link import LinkConfig, SimulatedLink
from sim.sensor import SensorConfig, SensorModel
from sim.tracker import ConstantVelocityTracker
from sim.investigation import run_sweep

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo" / "index.html"
MAX_RUNTIME_STEPS = 2000
S2_REPORT = ROOT / "runs" / "s2-physics" / "convergence.json"
S2_TELEMETRY = ROOT / "runs" / "s0-baseline" / "telemetry.csv"
S2_MANIFEST = ROOT / "runs" / "s0-baseline" / "manifest.json"
S7_RUN = ROOT / "runs" / "s7-swarm" / "run.json"
S7_MANIFEST = ROOT / "runs" / "s7-swarm" / "manifest.json"

def _link_fixture():
    """Return one retained, deterministic S3 link replay.

    This is deliberately a synthetic fixture. It exercises loss, delay,
    duplicate, reordering, and serialized contention fields without implying
    measured radio behavior.
    """
    config = LinkConfig(delay_s=0.12, delay_jitter_s=0.08,
                        loss_model="independent", loss_probability=0.2,
                        contention_mode="serialized", packet_duration_s=0.03)
    link = SimulatedLink(config, seed=1)
    for seq, send_time in ((0, 0.0), (1, 0.01), (2, 0.02), (2, 0.03),
                           (3, 0.04), (4, 0.05), (5, 0.06), (6, 0.07)):
        link.send("D1", "GCS", seq, send_time)
    link.deliver_all()
    return {"schema": "zephyr-s3-link-fixture-1", "status": "replay only; synthetic link fixture",
            "config": {"delay_s": config.delay_s, "delay_jitter_s": config.delay_jitter_s,
                       "loss_model": config.loss_model, "loss_probability": config.loss_probability,
                       "contention_mode": config.contention_mode,
                       "packet_duration_s": config.packet_duration_s},
            "events": [event.as_dict() for event in link.events],
            "unavailable": ["live radio", "manual controller", "swarm command", "measured link quality"]}

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _retained_telemetry():
    with S2_TELEMETRY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}

def _payload(telemetry, report=None, report_error=None):
    arrays = {k: [float(x) for x in v] for k, v in telemetry.items()}
    if report is None:
        report = {"schema": "zephyr-s2-report-1", "status": "frozen historical S2 verification fixture; report unavailable",
                  "checksum_scope": "full retained source files, not truncated payload arrays",
                  "payload": {"sample_count": len(next(iter(telemetry.values()), [])), "source_sample_count": None, "truncated": None},
                  "source": {"telemetry_path": "runs/s0-baseline/telemetry.csv", "telemetry_sha256": _sha256(S2_TELEMETRY)},
                  "unavailable_metrics": ["radio latency/loss", "sensor noise", "recovery"]}
    return {"schema": "zephyr-s2-demo-1", "label": "Synthetic single-drone S2 baseline replay",
            "status": "replay only; no live radio, camera, controller, or swarm", "telemetry": arrays,
            "report": report, "report_error": report_error}

def _report(telemetry):
    """Describe the payload separately from checksums of the full frozen source."""
    try:
        convergence = json.loads(S2_REPORT.read_text())
        manifest = json.loads(S2_MANIFEST.read_text())
        source_count = len(_retained_telemetry()["time"])
        payload_count = len(telemetry["time"])
        return {
            "schema": "zephyr-s2-report-1",
            "status": "frozen historical S2 verification fixture; not current-worktree provenance or flight performance",
            "checksum_scope": "full retained source files, not truncated payload arrays",
            "payload": {"sample_count": payload_count, "source_sample_count": source_count,
                        "truncated": payload_count < source_count},
            "provenance": {"artifact_status": manifest.get("artifact_status", "historical fixture"),
                           "source_git_revision": manifest.get("git_revision"),
                           "source_git_tree_state": manifest.get("git_tree_state"),
                           "current_worktree_match": "not claimed"},
            "source": {
                "report_path": "runs/s2-physics/convergence.json",
                "report_sha256": _sha256(S2_REPORT),
                "telemetry_path": "runs/s0-baseline/telemetry.csv",
                "telemetry_sha256": _sha256(S2_TELEMETRY),
                "manifest_path": "runs/s0-baseline/manifest.json",
                "manifest_sha256": _sha256(S2_MANIFEST),
            },
            "fixture": convergence,
            "unavailable_metrics": ["radio latency/loss", "sensor noise", "recovery"],
        }
    except Exception as exc:
        return {"schema": "zephyr-s2-report-error-1", "status": "report unavailable", "error": "report unavailable", "detail": str(exc)}

def _sensor_fixture():
    """Deterministic S4 truth/measurement fixture; it is not camera data."""
    model = SensorModel(SensorConfig(noise_std_m=0.02, bias_m=(0.01, -0.01, 0.0),
                                     dropout_probability=0.15), seed=7)
    rows = model.sample_series(((0.7 * i * 0.1, 0.0, 2.0) for i in range(40)),
                               (i * 0.1 for i in range(40)))
    return {"schema": "zephyr-s4-sensor-fixture-1",
            "status": "synthetic sensing fixture; no camera or flight measurement",
            "config": {"noise_std_m": 0.02, "bias_m": [0.01, -0.01, 0.0],
                       "dropout_probability": 0.15, "seed": 7},
            "samples": [row.as_dict() for row in rows]}

def _tracker_fixture():
    """Deterministic S5 constant-velocity recovery fixture."""
    sensor = SensorModel(SensorConfig(noise_std_m=0.02, dropout_probability=0.15), seed=9)
    tracker = ConstantVelocityTracker(position_m=(0, 0, 2), velocity_mps=(0.0, 0, 0),
                                       measurement_std_m=0.02)
    rows = []
    for i in range(80):
        t = i * 0.05
        truth = np.array((0.7 * t, 0.0, 2.0))
        sample = sensor.sample(truth, t if t > 0 else 1e-9)
        if sample.available:
            tracker.update(sample.measurement_m, sample.timestamp_s)
        state = tracker.predict(t if t > tracker.timestamp_s else tracker.timestamp_s)
        rows.append({"timestamp_s": t, "truth_m": truth.tolist(),
                     "measurement_m": None if not sample.available else list(sample.measurement_m),
                     "estimate_m": list(state.position_m), "available": sample.available,
                     "error_m": float(np.linalg.norm(np.asarray(state.position_m) - truth))})
    return {"schema": "zephyr-s5-tracker-fixture-1",
            "status": "synthetic tracker fixture; not controller recovery or flight performance",
            "config": {"model": "constant_velocity_kalman", "seed": 9, "noise_std_m": 0.02,
                       "dropout_probability": 0.15}, "samples": rows}

def _swarm_fixture():
    """Return the checked-in deterministic S7 event-log projection."""
    if not S7_RUN.exists() or not S7_MANIFEST.exists():
        return {"schema": "zephyr-s7-swarm-fixture-error-1", "status": "fixture unavailable"}
    run = json.loads(S7_RUN.read_text())
    manifest = json.loads(S7_MANIFEST.read_text())
    return {
        "schema": "zephyr-s7-swarm-fixture-1",
        "status": "synthetic deterministic replay; not flight performance",
        "run": run,
        "manifest": manifest,
        "source": {
            "run_path": "runs/s7-swarm/run.json",
            "run_sha256": _sha256(S7_RUN),
            "manifest_path": "runs/s7-swarm/manifest.json",
            "manifest_sha256": _sha256(S7_MANIFEST),
        },
        "unavailable": ["live radio", "camera", "manual authority", "flight performance"],
    }

class Handler(BaseHTTPRequestHandler):
    server_version = "ZephyrS2/1.0"
    def _send(self, code, body, ctype="application/json; charset=utf-8", disposition=None):
        if isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        if disposition: self.send_header("Content-Disposition", disposition)
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlsplit(self.path)
        if u.path == "/": return self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/run":
            q = parse_qs(u.query)
            if set(q) - {"limit"}: return self._send(400, json.dumps({"error": "only limit is allowed"}))
            try: limit = int(q.get("limit", [str(MAX_RUNTIME_STEPS)])[0])
            except (TypeError, ValueError): return self._send(400, json.dumps({"error": "limit must be an integer"}))
            if not 1 <= limit <= MAX_RUNTIME_STEPS: return self._send(400, json.dumps({"error": "limit out of range"}))
            t = _retained_telemetry(); t = {k: v[:limit] for k, v in t.items()}
            return self._send(200, json.dumps(_payload(t, _report(t))))
        if u.path == "/api/link":
            if u.query:
                return self._send(400, json.dumps({"error": "query parameters are not allowed"}))
            return self._send(200, json.dumps(_link_fixture()))
        if u.path == "/api/sensor":
            if u.query: return self._send(400, json.dumps({"error": "query parameters are not allowed"}))
            return self._send(200, json.dumps(_sensor_fixture()))
        if u.path == "/api/tracker":
            if u.query: return self._send(400, json.dumps({"error": "query parameters are not allowed"}))
            return self._send(200, json.dumps(_tracker_fixture()))
        if u.path == "/api/investigation":
            if u.query: return self._send(400, json.dumps({"error": "query parameters are not allowed"}))
            return self._send(200, json.dumps(run_sweep()))
        if u.path == "/api/swarm":
            if u.query:
                return self._send(400, json.dumps({"error": "query parameters are not allowed"}))
            return self._send(200, json.dumps(_swarm_fixture()))
        if u.path in ("/download/telemetry.csv", "/download/run.json"):
            t = _retained_telemetry()
            if u.path.endswith("csv"):
                out = io.StringIO(); w = csv.writer(out); names = tuple(t); w.writerow(names); w.writerows(zip(*(t[n] for n in names)))
                return self._send(200, out.getvalue(), "text/csv; charset=utf-8", 'attachment; filename="telemetry.csv"')
            return self._send(200, json.dumps(_payload(t, _report(t)), indent=2), disposition='attachment; filename="run.json"')
        return self._send(404, json.dumps({"error": "not found"}))
    def do_POST(self):
        u = urlsplit(self.path)
        if u.path != "/api/rerun": return self._send(404, json.dumps({"error": "not found"}))
        if u.query: return self._send(400, json.dumps({"error": "query parameters are not allowed on POST /api/rerun"}))
        try: length = int(self.headers.get("Content-Length", "0"))
        except ValueError: return self._send(400, json.dumps({"error": "invalid content length"}))
        if length < 0 or length > 1024 * 1024: return self._send(400, json.dumps({"error": "request too large"}))
        raw = self.rfile.read(length)
        if raw.strip() not in (b"", b"{}"):
            return self._send(400, json.dumps({"error": "rerun accepts an empty JSON object only"}))
        t = _retained_telemetry()
        return self._send(200, json.dumps(_payload(t, _report(t))))
    def log_message(self, *_): pass

def serve(host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Zephyr S2 demo: http://{host}:{port}", flush=True); httpd.serve_forever()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Serve the offline Zephyr S2 physics replay")
    p.add_argument("--host", default="127.0.0.1", help="explicit local bind address")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(); serve(args.host, args.port)
