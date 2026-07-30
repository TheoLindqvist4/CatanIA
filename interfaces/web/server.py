"""A local web server for the game.

    python -m interfaces.web            then open http://127.0.0.1:8000

Deliberately the standard library and nothing else. A single-player local game does not
need a framework, and adding one would mean a dependency, an install and a version to pin
for something this thin.

All the thinking lives in :mod:`interfaces.web.api`, which knows nothing about HTTP. This
module maps URLs onto it and serves files. Swapping in FastAPI later would rewrite this file
and touch nothing else.

Routes
------
``GET  /``                        the page
``GET  /app/<file>``              client code
``GET  /images/<path>``           board art
``GET  /api/geometry``            board coordinates — static, fetched once
``POST /api/game``                start one: ``{opponent, rules, seed}``
``GET  /api/game/<id>``           the current position, as the human may see it
``POST /api/game/<id>/action``    play: ``{"index": n}``
"""

import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from interfaces.web import api

HERE = pathlib.Path(__file__).parent
CLIENT = HERE / "static"
IMAGES = HERE.parent / "static" / "images"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
}

GAMES = api.Games()


def safe_path(root, relative):
    """``root / relative``, or ``None`` if it escapes ``root``.

    A served path arrives from the network, so ``..`` has to be refused rather than
    trusted — otherwise the whole filesystem is readable.
    """
    try:
        target = (root / relative.lstrip("/")).resolve()
    except (OSError, ValueError):
        return None
    if root.resolve() not in target.parents and target != root.resolve():
        return None
    return target if target.is_file() else None


class Handler(BaseHTTPRequestHandler):
    server_version = "CatanIA"

    # ---------------------------------------------------------------- #

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._send_file(CLIENT / "index.html")
            if path.startswith("/app/"):
                return self._send_static(CLIENT, path[len("/app/"):])
            if path.startswith("/images/"):
                return self._send_static(IMAGES, path[len("/images/"):])
            if path == "/api/geometry":
                return self._send_json(api.geometry())
            if path.startswith("/api/game/"):
                return self._send_json(GAMES.get(path[len("/api/game/"):]).view())
        except KeyError as error:
            return self._send_json({"error": str(error)}, status=404)
        self._send_json({"error": f"no route for {path}"}, status=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json()
            if path == "/api/game":
                game = GAMES.new(
                    opponent=body.get("opponent", "greedy"),
                    rules_name=body.get("rules", "ranked1v1"),
                    seed=body.get("seed"),
                )
                return self._send_json(game.view())
            if path.startswith("/api/game/") and path.endswith("/action"):
                game_id = path[len("/api/game/"):-len("/action")]
                index = body.get("index")
                if not isinstance(index, int):
                    raise ValueError("expected an integer 'index'")
                return self._send_json(GAMES.get(game_id).play(index))
        except KeyError as error:
            return self._send_json({"error": str(error)}, status=404)
        except ValueError as error:
            # An illegal or out-of-turn action. The client should not have offered it, so
            # say what happened rather than failing silently.
            return self._send_json({"error": str(error)}, status=400)
        self._send_json({"error": f"no route for {path}"}, status=404)

    # ---------------------------------------------------------------- #

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            raise ValueError("body was not JSON") from None

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, root, relative):
        target = safe_path(root, relative)
        if target is None:
            return self._send_json({"error": "not found"}, status=404)
        self._send_file(target)

    def _send_file(self, target):
        if not target.is_file():
            return self._send_json({"error": "not found"}, status=404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        """Quieter than the default, which logs every image request."""
        if "/api/" in str(args[0] if args else ""):
            sys.stderr.write(f"  {fmt % args}\n")


def serve(host="127.0.0.1", port=8000):
    """Run until interrupted. Bound to localhost: this has no authentication."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"CatanIA — open http://{host}:{port}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
