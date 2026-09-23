"""Local review server for the same frontend/controller, using only stdlib.

python -m frontend.preview --port 8765
This is a single-user local preview, not a production deployment.
"""
from __future__ import annotations

import argparse
import base64
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from threading import RLock

from frontend.controller import Controller

ASSETS = Path(__file__).parent / "assets"
HTML = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TwinTurbo.ai · Wind Intelligence</title><link rel="icon" href="data:,"><link rel="stylesheet" href="/dashboard.css"><style>body{margin:0;background:#101513}</style></head><body><div id="twinturbo-app"></div><script type="module">
import {mountDashboard} from '/dashboard.js';
const root=document.querySelector('#twinturbo-app');
async function send(action){const response=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action)});if(!response.ok)throw new Error('Request failed');mountDashboard(root,await response.json(),send);}
const response=await fetch('/api/state');mountDashboard(root,await response.json(),send);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    sessions = {}
    lock = RLock()

    def response(self, body, mime, status=200, cookie=None, filename=None):
        body = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        if cookie:
            self.send_header("Set-Cookie", f"tt_session={cookie}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def get_controller(self):
        cookies = SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
        except Exception:
            pass
        identity = cookies.get("tt_session")
        identity = identity.value if identity else None
        if identity not in self.sessions:
            identity = secrets.token_urlsafe(24)
            self.sessions[identity] = Controller()
        return identity, self.sessions[identity]

    def do_GET(self):
        if self.path == "/":
            return self.response(HTML, "text/html; charset=utf-8")
        if self.path in ("/dashboard.css", "/dashboard.js"):
            mime = "text/css" if self.path.endswith("css") else "text/javascript"
            return self.response((ASSETS / self.path[1:]).read_bytes(), mime + "; charset=utf-8")
        if self.path == "/api/state":
            with self.lock:
                identity, controller = self.get_controller()
                return self.response(json.dumps(controller.view(), ensure_ascii=False, allow_nan=False), "application/json; charset=utf-8", cookie=identity)
        if self.path == "/api/download":
            with self.lock:
                identity, controller = self.get_controller()
                artifact = controller.download
                if not artifact:
                    return self.response("No ready export", "text/plain", 404)
                return self.response(base64.b64decode(artifact["base64"]), "text/csv; charset=utf-8", filename=artifact["filename"])
        return self.response("Not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/action":
            return self.response("Not found", "text/plain", 404)
        expected = f"http://{self.headers.get('Host')}"
        if self.headers.get("Origin") not in (None, expected) or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.response("Forbidden", "text/plain", 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 8192:
                raise ValueError()
            action = json.loads(self.rfile.read(size))
            if not isinstance(action, dict):
                raise ValueError()
        except (ValueError, json.JSONDecodeError):
            return self.response("Invalid action", "text/plain", 400)
        with self.lock:
            identity, controller = self.get_controller()
            state = controller.dispatch(action)
            if state.get("download"):
                state["download"] = {**state["download"], "url": "/api/download"}
            self.response(json.dumps(state, ensure_ascii=False, allow_nan=False), "application/json; charset=utf-8", cookie=identity)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"TwinTurbo.ai local preview: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
