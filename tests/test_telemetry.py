from types import SimpleNamespace
import os
import json
import unittest
from unittest.mock import MagicMock, patch

from utils.telemetry import Telemetry, mask_otel_spans, trace_stage, llm_trace_config


class TelemetryTests(unittest.TestCase):
    def test_disabled_and_missing_keys(self):
        with patch.dict(os.environ, {"LANGFUSE_TRACING_ENABLED": "false"}):
            self.assertIsNone(Telemetry.from_environment().client)
        with patch.dict(os.environ, {"LANGFUSE_TRACING_ENABLED": "true",
                                   "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}):
            with self.assertLogs("utils.telemetry", level="WARNING"):
                self.assertIsNone(Telemetry.from_environment().client)

    def test_span_failure_does_not_change_result(self):
        client = MagicMock()
        client.start_as_current_observation.side_effect = RuntimeError("offline")
        with self.assertLogs("utils.telemetry", level="WARNING"):
            with Telemetry(client).span("test") as observation:
                self.assertIsNone(observation)

    def test_pipeline_error_is_propagated_and_observation_marked(self):
        client = MagicMock()
        manager = client.start_as_current_observation.return_value
        observation = manager.__enter__.return_value
        with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
            with Telemetry(client).span("test"):
                raise RuntimeError("pipeline failed")
        observation.update.assert_called_once_with(level="ERROR", status_message="Pipeline step failed")
        manager.__exit__.assert_called_once()

    def test_finish_and_update_failures_are_nonfatal(self):
        client = MagicMock()
        manager = client.start_as_current_observation.return_value
        observation = manager.__enter__.return_value
        observation.update.side_effect = RuntimeError("offline")
        manager.__exit__.side_effect = RuntimeError("offline")
        with self.assertLogs("utils.telemetry", level="WARNING"):
            with Telemetry(client).span("test") as span:
                Telemetry.update(span, output="answer")

    def test_stage_returns_identical_result_with_or_without_tracing(self):
        class Retriever:
            @trace_stage("retrieve-context", as_type="retriever")
            def retrieve(self, query):
                return [{"id": "1", "score": .8, "content": "private payload"}]
        owner = Retriever()
        expected = owner.retrieve("q")
        client = MagicMock()
        owner.telemetry = Telemetry(client)
        self.assertEqual(owner.retrieve("q"), expected)
        client.start_as_current_observation.assert_called_once_with(
            name="retrieve-context", as_type="retriever", input="q")
        client.start_as_current_observation.return_value.__enter__.return_value.update.assert_called_once_with(
            output=[{"id": "1", "score": .8}])

    def test_masking_handles_pii_and_keys(self):
        params = SimpleNamespace(spans={"span": SimpleNamespace(attributes={
            "langfuse.observation.input": '"Email alice@example.com, phone 0912345678, ID 123456789012, sk-lf-test-key"',
            "gen_ai.request.model": "qwen3.5:9b"})})
        with patch.dict(os.environ, {"LANGFUSE_CAPTURE_CONTENT": "true"}):
            result = mask_otel_spans(params=params)
        masked = result.span_patches["span"].set_attributes["langfuse.observation.input"]
        for value in ("alice@example.com", "0912345678", "123456789012", "sk-lf-test-key"):
            self.assertNotIn(value, masked)
        self.assertNotIn("gen_ai.request.model", result.span_patches["span"].set_attributes)

    def test_content_can_be_disabled(self):
        params = SimpleNamespace(spans={"span": SimpleNamespace(attributes={
            "langfuse.observation.input": '"question"', "langfuse.observation.output": '"answer"'})})
        with patch.dict(os.environ, {"LANGFUSE_CAPTURE_CONTENT": "false"}):
            result = mask_otel_spans(params=params)
        self.assertTrue(all(value == '"[CONTENT DISABLED]"'
                            for value in result.span_patches["span"].set_attributes.values()))

    def test_masking_preserves_json_and_numeric_scores(self):
        value = {"email": "alice@example.com", "score": 0.123456789012}
        params = SimpleNamespace(spans={"span": SimpleNamespace(attributes={
            "langfuse.observation.input": json.dumps(value)})})
        with patch.dict(os.environ, {"LANGFUSE_CAPTURE_CONTENT": "true"}):
            result = mask_otel_spans(params=params)
        cleaned = json.loads(result.span_patches["span"].set_attributes["langfuse.observation.input"])
        self.assertEqual(cleaned["email"], "[EMAIL]")
        self.assertEqual(cleaned["score"], value["score"])

    def test_callbacks_only_when_enabled(self):
        self.assertEqual(llm_trace_config(SimpleNamespace()), {})
        owner = SimpleNamespace(telemetry=Telemetry(MagicMock()))
        with patch("langfuse.langchain.CallbackHandler") as handler:
            self.assertEqual(llm_trace_config(owner), {"config": {"callbacks": [handler.return_value]}})

    def test_shutdown_flushes_and_is_nonfatal(self):
        client = MagicMock()
        Telemetry(client).shutdown()
        client.shutdown.assert_called_once()
        client.shutdown.side_effect = RuntimeError("offline")
        with self.assertLogs("utils.telemetry", level="WARNING"):
            Telemetry(client).shutdown()


if __name__ == "__main__":
    unittest.main()
