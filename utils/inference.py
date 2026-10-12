"""Shared LangChain inference selection for generation and legal detection."""
from urllib.parse import urlparse

import requests

from config import Config


def _deepseek_settings():
    url = Config.DEEPSEEK_BASE_URL.rstrip("/")
    parsed = urlparse(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("DEEPSEEK_BASE_URL must be an HTTPS URL without credentials or query")
    if not Config.DEEPSEEK_API_KEY:
        raise ValueError("Set DEEPSEEK_API_KEY in the project's .env file")
    return url, Config.DEEPSEEK_API_KEY


def create_chat_model():
    if Config.INFERENCE_PROVIDER == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        url, key = _deepseek_settings()
        return ChatDeepSeek(
            model=Config.MODEL_GEN, api_key=key, base_url=url,
            temperature=0.1, max_tokens=Config.GENERATION_MAX_TOKENS,
            timeout=120, max_retries=2,
            extra_body={"thinking": {"type": "disabled"}},
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=Config.MODEL_GEN, base_url=Config.OLLAMA_BASE_URL,
        temperature=0.1, reasoning=False,
        num_ctx=Config.OLLAMA_CONTEXT, num_predict=Config.GENERATION_MAX_TOKENS,
    )


def check_inference_ready():
    """Validate authentication and model availability without generating tokens."""
    if Config.INFERENCE_PROVIDER == "deepseek":
        url, key = _deepseek_settings()
        response = requests.get(url + "/models", headers={"Authorization": "Bearer " + key}, timeout=5)
        response.raise_for_status()
        names = {model["id"] for model in response.json()["data"]}
        if Config.MODEL_GEN not in names:
            raise RuntimeError(f"Configured DeepSeek model is unavailable: {Config.MODEL_GEN}")
    else:
        response = requests.get(Config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags", timeout=5)
        response.raise_for_status()
        names = {model["name"] for model in response.json()["models"]}
        model = Config.MODEL_GEN if ":" in Config.MODEL_GEN else Config.MODEL_GEN + ":latest"
        if model not in names:
            raise RuntimeError(f"Download the configured Ollama model: {Config.MODEL_GEN}")
