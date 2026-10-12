import unittest
from unittest.mock import patch

from hf_space import serve


class SpaceStartupTests(unittest.TestCase):
    def test_requires_login_before_starting_backend(self):
        with patch.dict("os.environ", {"DEMO_USERNAME": "", "DEMO_PASSWORD": ""}), patch("hf_space.subprocess.Popen") as start:
            with self.assertRaises(RuntimeError):
                serve()
            start.assert_not_called()

    def test_ready_backend_launches_ui_and_is_cleaned_up(self):
        with patch.dict("os.environ", {"DEMO_USERNAME": "demo", "DEMO_PASSWORD": "test-password-123"}), patch("hf_space.subprocess.Popen") as start, patch("hf_space.urlopen"), patch("hf_space.main") as ui:
            start.return_value.poll.return_value = None
            serve()
            ui.assert_called_once_with()
            args = start.call_args.args[0]
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            start.return_value.terminate.assert_called_once_with()

    def test_failed_backend_does_not_launch_ui(self):
        with patch.dict("os.environ", {"DEMO_USERNAME": "demo", "DEMO_PASSWORD": "test-password-123"}), patch("hf_space.subprocess.Popen") as start, patch("hf_space.main") as ui:
            start.return_value.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                serve()
            ui.assert_not_called()
            start.return_value.terminate.assert_called_once_with()
