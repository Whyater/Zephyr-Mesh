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

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo" / "index.html"
MAX_RUNTIME_STEPS = 2000
S2_REPORT = ROOT / "runs" / "s2-physics" / "convergence.json"
S2_TELEMETRY = ROOT / "runs" / "s0-baseline" / "telemetry.csv"
S2_MANIFEST = ROOT / "runs" / "s0-baseline" / "manifest.json"

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _retained_telemetry():
    with S2_TELEMETRY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}

def _payload(telemetry, report=None, report_error=None):
    arrays = {k: [float(x) for x in v] for k, v in telemetry.items()}
    if report is None:
        report = {"schema": "zephyr-s2-report-1", "status": "retained S2 physics verification fixture; not flight performance",
                  "source": {"telemetry_path": "runs/s0-baseline/telemetry.csv", "telemetry_sha256": _sha256(S2_TELEMETRY)},
                  "unavailable_metrics": ["radio latency/loss", "sensor noise", "recovery"]}
    return {"schema": "zephyr-s2-demo-1", "label": "Synthetic single-drone S2 baseline replay",
            "status": "replay only; no live radio, camera, controller, or swarm", "telemetry": arrays,
            "report": report, "report_error": report_error}

def _report(telemetry):
    """Return the retained S2 verification report with durable source checksums."""
    try:
        convergence = json.loads(S2_REPORT.read_text())
        return {
            "schema": "zephyr-s2-report-1",
            "status": "retained S2 physics verification fixture; not flight performance",
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
    p.add_argument("--host", default="127.0.0.1", help="bind address; use a Tailscale IP explicitly for phone access")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(); serve(args.host, args.port)
