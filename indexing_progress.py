"""Read-only localhost progress view for the hosted corpus indexer."""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from string import Template

import requests


def process_running(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def tail(path):
    with Path(path).open("rb") as handle:
        handle.seek(max(0, handle.seek(0, 2) - 16000))
        return handle.read().decode("utf-8", errors="replace")


def snapshot(results, total, qdrant):
    count, status, eta = None, "No active run", "—"
    try:
        response = requests.post(
            qdrant.rstrip("/") + "/collections/hosted_baai_bge_m3_1024/points/count",
            json={"exact": True}, timeout=5,
        )
        response.raise_for_status()
        count = response.json()["result"]["count"]
    except (requests.RequestException, ValueError, KeyError):
        pass
    try:
        active = json.loads((results / "hosted-bge-m3-active-run.json").read_text())
        stdout, stderr = tail(active["stdout"]), tail(active["stderr"])
        running = process_running(int(active["pid"]))
        status = "Indexing" if running else "Stopped"
        if not running and "RuntimeError" in stderr:
            status = "Interrupted"
        estimates = re.findall(r"<([\d:]+),", stderr)
        if running and estimates:
            parts = [int(part) for part in estimates[-1].split(":")]
            seconds = sum(value * 60 ** i for i, value in enumerate(reversed(parts)))
            hours, minutes = divmod(seconds // 60, 60)
            eta = f"{hours}h {minutes}m" if hours else f"{minutes}m"
        if count == total and "Verified complete hosted index" in stdout:
            status, eta = "Complete", "Finished"
    except (OSError, ValueError, KeyError):
        pass
    return {"count": count, "total": total, "status": status, "eta": eta}


def render(data):
    count, total = data["count"], data["total"]
    percent = min(100, count / total * 100) if count is not None else 0
    return Template(Path(__file__).with_name("indexing_progress.html").read_text(encoding="utf-8")).substitute(
        count=f"{count:,}" if count is not None else "Unavailable",
        total=f"{total:,}", remaining=f"{max(0, total-count):,}" if count is not None else "—",
        percent=f"{percent:.1f}", percent_label=f"{percent:.1f}% complete" if count is not None else "Waiting for Qdrant",
        status=escape(data["status"]), eta=escape(data["eta"]),
        checked=datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S"),
        notice="Saved article count could not be read. Check that Docker and Qdrant are running." if count is None else "",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/status"):
                self.send_error(404)
                return
            data = snapshot(args.results, 61068, args.qdrant)
            body = (json.dumps(data) if self.path == "/status" else render(data)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if self.path == "/status" else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"Progress view: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
