import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from indexing_progress import render, snapshot


class ProgressTests(unittest.TestCase):
    def test_running_and_stopped_status_use_live_count_and_hide_stale_eta(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, err = root / "out", root / "err"
            out.write_text("Resume: saved batches skipped")
            err.write_text("Indexing articles: 20% [18:13<1:45:42, 5.01s/it]")
            (root / "hosted-bge-m3-active-run.json").write_text(json.dumps(
                {"pid": 123, "stdout": str(out), "stderr": str(err)}))
            response = MagicMock()
            response.json.return_value = {"result": {"count": 40832}}
            with patch("indexing_progress.requests.post", return_value=response), patch("indexing_progress.process_running", return_value=True):
                data = snapshot(root, 61068, "http://localhost:6333")
                self.assertEqual(data["status"], "Indexing")
                self.assertEqual(data["eta"], "1h 45m")
                self.assertIn("40,832", render(data))
            with patch("indexing_progress.requests.post", return_value=response), patch("indexing_progress.process_running", return_value=False):
                data = snapshot(root, 61068, "http://localhost:6333")
                self.assertEqual(data["status"], "Stopped")
                self.assertEqual(data["eta"], "—")

    def test_verified_complete_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, err = root / "out", root / "err"
            out.write_text("Verified complete hosted index: 61,068 articles.")
            err.write_text("")
            (root / "hosted-bge-m3-active-run.json").write_text(json.dumps(
                {"pid": 123, "stdout": str(out), "stderr": str(err)}))
            response = MagicMock()
            response.json.return_value = {"result": {"count": 61068}}
            with patch("indexing_progress.requests.post", return_value=response), patch("indexing_progress.process_running", return_value=False):
                self.assertEqual(snapshot(root, 61068, "http://localhost:6333")["status"], "Complete")
