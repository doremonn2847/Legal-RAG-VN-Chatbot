"""Run the existing private FastAPI backend and authenticated Gradio UI on Spaces."""
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from app import main


def serve():
    os.environ["API_BASE_URL"] = "http://127.0.0.1:8000"
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SHARE"] = "false"
    if not os.getenv("DEMO_USERNAME") or len(os.getenv("DEMO_PASSWORD", "")) < 12:
        raise RuntimeError("Set demo login secrets before publishing the Space")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"],
        cwd=Path(__file__).resolve().parent,
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if backend.poll() is not None:
                raise RuntimeError("FastAPI startup failed; check the Space logs")
            try:
                with urlopen("http://127.0.0.1:8000/health", timeout=2):
                    break
            except (URLError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("FastAPI startup exceeded ten minutes")
        main()
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait()


if __name__ == "__main__":
    serve()
