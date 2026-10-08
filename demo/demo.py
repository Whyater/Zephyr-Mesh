"""Small local Zephyr S1 replay server. Serves only the bundled browser UI and baseline artifacts."""
from __future__ import annotations
import argparse, csv, io, json, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import numpy as np
from sim.baseline import BaselineConfig, run_baseline, write_baseline

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo" / "index.html"
MAX_RUNTIME_STEPS = 2000

def _payload(telemetry, report=None, report_error=None):
    arrays = {k: [float(x) for x in v] for k,v in telemetry.items()}
    if report is None:
        z = np.asarray(telemetry["z"]); t = np.asarray(telemetry["time"])
        report = {"schema":"zephyr-s1-report-1", "tracking_error":{"count":len(z)},
                  "step_response":{"rise_time_s":None,"overshoot_pct":None,"settling_time_s":None},
                  "unavailable_metrics":["radio latency/loss","sensor noise","recovery"],
                  "known_issues":["Integrator labeled RK4 behaves as forward Euler","Drag is not applied"]}
    return {"schema":"zephyr-s1-demo-1", "label":"Synthetic single-drone baseline replay",
            "status":"synthetic; no simulated/measured link, sensor, or recovery", "telemetry":arrays,
            "report":report, "report_error": report_error}

def _report(telemetry):
    try:
        from sim.report import build_baseline_report
        with tempfile.TemporaryDirectory() as d:
            write_baseline(d, telemetry=telemetry)
            return build_baseline_report(Path(d))
    except Exception as exc:
        return {"schema":"zephyr-s1-report-error-1", "error":"report unavailable", "detail":str(exc)}

class Handler(BaseHTTPRequestHandler):
    server_version = "ZephyrS1/1.0"
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
            if set(q) - {"limit"}: return self._send(400, json.dumps({"error":"only limit is allowed"}))
            try: limit = int(q.get("limit", ["2000"])[0])
            except ValueError: return self._send(400, json.dumps({"error":"limit must be an integer"}))
            if not 1 <= limit <= MAX_RUNTIME_STEPS: return self._send(400, json.dumps({"error":"limit out of range"}))
            t = run_baseline(BaselineConfig())
            t = {k:v[:limit] for k,v in t.items()}
            return self._send(200, json.dumps(_payload(t, _report(t))))
        if u.path in ("/download/telemetry.csv", "/download/run.json"):
            t = run_baseline(BaselineConfig())
            if u.path.endswith("csv"):
                out=io.StringIO(); w=csv.writer(out); names=tuple(t); w.writerow(names); w.writerows(zip(*(t[n] for n in names)))
                return self._send(200, out.getvalue(), "text/csv; charset=utf-8", 'attachment; filename="telemetry.csv"')
            return self._send(200, json.dumps(_payload(t, _report(t)), indent=2), disposition='attachment; filename="run.json"')
        return self._send(404, json.dumps({"error":"not found"}))
    def do_POST(self):
        u=urlsplit(self.path)
        if u.path != "/api/rerun": return self._send(404, json.dumps({"error":"not found"}))
        length=int(self.headers.get("Content-Length", "0"))
        if length > 0:
            raw=self.rfile.read(length)
            if raw.strip() not in (b"", b"{}"): return self._send(400, json.dumps({"error":"rerun accepts an empty JSON object only"}))
        t=run_baseline(BaselineConfig())
        return self._send(200, json.dumps(_payload(t, _report(t))))
    def log_message(self, *_): pass

def serve(host="127.0.0.1", port=8765):
    httpd=ThreadingHTTPServer((host,port), Handler)
    print(f"Zephyr S1 demo: http://{host}:{port}", flush=True); httpd.serve_forever()

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Serve the offline Zephyr S1 baseline replay")
    p.add_argument("--host", default="127.0.0.1", help="bind address; use a Tailscale IP explicitly for phone access")
    p.add_argument("--port", type=int, default=8765)
    args=p.parse_args(); serve(args.host,args.port)
