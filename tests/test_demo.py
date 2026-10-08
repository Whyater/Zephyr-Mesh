import json, threading, urllib.request, urllib.error
from demo.demo import Handler, ThreadingHTTPServer

def request(server, path, method="GET", body=None):
    req=urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", data=body, method=method)
    return urllib.request.urlopen(req)

def test_allowlisted_endpoints_and_rejections():
    s=ThreadingHTTPServer(("127.0.0.1",0),Handler); threading.Thread(target=s.serve_forever,daemon=True).start()
    try:
        assert request(s,"/").status == 200
        data=json.load(request(s,"/api/run?limit=3")); assert len(data["telemetry"]["time"]) == 3
        assert request(s,"/download/telemetry.csv").headers["Content-Disposition"]
        assert request(s,"/download/run.json").status == 200
        for path in ("/private", "/api/run?wat=1", "/api/run?limit=no"):
            try: request(s,path); assert False
            except urllib.error.HTTPError as e: assert e.code in (400,404)
    finally: s.shutdown()

def test_rerun_rejects_arbitrary_input():
    s=ThreadingHTTPServer(("127.0.0.1",0),Handler); threading.Thread(target=s.serve_forever,daemon=True).start()
    try:
        try: request(s,"/api/rerun","POST",b'{"dt":0.00001}') ; assert False
        except urllib.error.HTTPError as e: assert e.code == 400
    finally: s.shutdown()


def test_ui_states_tailscale_trust_boundary():
    from pathlib import Path
    html = Path("demo/index.html").read_text()
    assert "unauthenticated replay only to trusted peers" in html
    assert "loopback is the default" in html
