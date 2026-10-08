"""Opt-in Langfuse tracing; telemetry failures never change inference results."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import logging
import json
import os
import re

logger = logging.getLogger(__name__)

PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+84|0)(?:[ .-]?\d){9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{12}(?!\d)"), "[ID]"),
    (re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9-]+"), "[KEY]"),
]


def mask_text(value):
    for pattern, replacement in PII_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def mask_data(value):
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, list):
        return [mask_data(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_data(item) for key, item in value.items()}
    return value


def mask_otel_spans(*, params):
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
    patches = {}
    capture = os.getenv("LANGFUSE_CAPTURE_CONTENT", "true").lower() == "true"
    for identifier, span in params.spans.items():
        replacements = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                masked = value
                if not capture and (key.endswith(".input") or key.endswith(".output")):
                    masked = '"[CONTENT DISABLED]"'
                else:
                    try:
                        data = json.loads(value)
                    except (ValueError, TypeError):
                        masked = mask_text(value)
                    else:
                        cleaned = mask_data(data)
                        if cleaned != data:
                            masked = json.dumps(cleaned, ensure_ascii=False)
                if masked != value:
                    replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches)


class Telemetry:
    def __init__(self, client=None):
        self.client = client

    @classmethod
    def from_environment(cls):
        if os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower() != "true":
            return cls()
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        if not public or not secret:
            logger.warning("Langfuse tracing disabled: project keys are missing")
            return cls()
        try:
            from langfuse import Langfuse
            return cls(Langfuse(public_key=public, secret_key=secret,
                               base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                               mask_otel_spans=mask_otel_spans,
                               environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "local")))
        except Exception:
            logger.warning("Langfuse initialization failed; tracing disabled")
            return cls()

    @contextmanager
    def span(self, name, session_id=None, **kwargs):
        manager, observation, attributes = None, None, None
        if self.client is not None:
            try:
                manager = self.client.start_as_current_observation(name=name, **kwargs)
                observation = manager.__enter__()
            except Exception:
                manager = None
                logger.warning("Could not start Langfuse observation")
        if observation is not None and session_id:
            try:
                from langfuse import propagate_attributes
                attributes = propagate_attributes(session_id=session_id, tags=["legal-rag"])
                attributes.__enter__()
            except Exception:
                attributes = None
                logger.warning("Could not attach Langfuse conversation ID")
        try:
            yield observation
        except BaseException:
            self.update(observation, level="ERROR", status_message="Pipeline step failed")
            raise
        finally:
            if attributes is not None:
                try:
                    attributes.__exit__(None, None, None)
                except Exception:
                    logger.warning("Could not finish Langfuse attributes")
            if manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception:
                    logger.warning("Could not finish Langfuse observation")

    @staticmethod
    def update(observation, **kwargs):
        if observation is not None:
            try:
                observation.update(**kwargs)
            except Exception:
                logger.warning("Could not update Langfuse observation")

    def shutdown(self):
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:
                logger.warning("Could not flush Langfuse observations on shutdown")


def trace_stage(name, as_type="span"):
    def decorate(function):
        @wraps(function)
        def wrapped(self, query, *args, **kwargs):
            telemetry = getattr(self, "telemetry", None)
            stage_input = query
            if telemetry and args and isinstance(args[0], list):
                stage_input = {"question": query, "candidates": [
                    {"id": doc.get("id"), "score": doc.get("score"),
                     "excerpt": doc.get("content", "")[:1000]} for doc in args[0]]}
            context = telemetry.span(name, as_type=as_type, input=stage_input) if telemetry else nullcontext()
            with context as observation:
                result = function(self, query, *args, **kwargs)
                if telemetry:
                    if isinstance(result, list):
                        output = [{"id": doc.get("id"), "score": doc.get("score")}
                                  for doc in result]
                    else:
                        output = result
                    telemetry.update(observation, output=output)
                return result
        return wrapped
    return decorate


def llm_trace_config(owner):
    telemetry = getattr(owner, "telemetry", None)
    if telemetry is None or telemetry.client is None:
        return {}
    try:
        from langfuse.langchain import CallbackHandler
        return {"config": {"callbacks": [CallbackHandler(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"))]}}
    except Exception:
        logger.warning("Could not create Langfuse LangChain callback")
        return {}
