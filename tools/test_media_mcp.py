"""Unit tests for tools/media_mcp.py lifecycle helpers, GPU selection, and alignment."""

from __future__ import annotations

import http.server
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import media_mcp  # noqa: E402


def main() -> None:
    calls: list[str] = []
    next_status = [200]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            calls.append(self.path)
            self.send_response(next_status[0])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if next_status[0] == 200:
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.wfile.write(b'{"error":"busy"}')

        def log_message(self, *args: object) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        os.environ["SHADOWZERO_SERVE_URL"] = f"http://127.0.0.1:{port}"

        with media_mcp._park_shadowzero():
            calls.append("worker")
        assert calls == ["/v1/engine/park", "worker", "/v1/engine/resume"], calls

        calls.clear()
        try:
            with media_mcp._park_shadowzero():
                calls.append("crash")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert calls == ["/v1/engine/park", "crash", "/v1/engine/resume"], calls

        calls.clear()
        next_status[0] = 500
        try:
            with media_mcp._park_shadowzero():
                calls.append("should-not-run")
            raise AssertionError("expected RuntimeError when park returns 500")
        except RuntimeError:
            pass
        assert calls == ["/v1/engine/park"], calls

        gpu = media_mcp._pick_best_gpu()
        assert gpu == "1", f"expected RTX 5060 Ti (1), got {gpu}"
        assert media_mcp._snap(1080, 16, 256, 2048) == 1072
        assert media_mcp._snap(720, 32, 256, 1280) == 704
        assert media_mcp._snap(1920, 16, 256, 2048) == 1920
        assert media_mcp._resolve_target_dims(1280, 720, 3840, 0, align=16) == (3840, 2160)
        assert media_mcp._resolve_target_dims(1280, 704, 3840, 2160, align=16) == (3840, 2160)
        assert media_mcp._tile_starts(1280, 384, 32) == [0, 352, 704, 896]
        assert media_mcp._tile_starts(256, 384, 32) == [0]
        print("ALL media_mcp unit tests passed (park/resume, error safety, GPU=1, alignment, tiling).")
    finally:
        os.environ.pop("SHADOWZERO_SERVE_URL", None)
        srv.shutdown()


if __name__ == "__main__":
    main()
