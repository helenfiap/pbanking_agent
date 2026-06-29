import os
import threading

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

AZURE_MODELS = {
    "azure":          {"heavy": "gpt-4o",                      "cheap": "gpt-4.1"},
    "azure-41":       {"heavy": "gpt-4.1",                     "cheap": "gpt-4.1"},
    "azure-kimi":     {"heavy": "Kimi-K2.5",                   "cheap": "Kimi-K2.5"},
    "azure-grok":     {"heavy": "grok-4-1-fast-non-reasoning", "cheap": "grok-4-1-fast-non-reasoning"},
    "azure-nano":     {"heavy": "gpt-5.4-nano",                "cheap": "gpt-5.4-nano"},
    "azure-deepseek": {"heavy": "DeepSeek-V3.2",               "cheap": "DeepSeek-V3.2"},
}

# gemini-2.5-flash:      best quality, 20 req/day free (default)
# gemini-2.0-flash-lite: 1500 req/day free (gemini-lite provider)
# gemini-2.5-pro:        thinking model, slow (gemini-pro provider)
GEMINI_PRIMARY  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK = "gemini-3.5-flash"

# Thread-local provider override for benchmark tab
_thread_local = threading.local()


def _make_gemini(model: str) -> ChatGoogleGenerativeAI:
    """Simplest possible Gemini instantiation — reads key directly from env."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_2")
    if not api_key:
        raise ValueError(
            "Nenhuma GOOGLE_API_KEY encontrada. "
            "Adicione GOOGLE_API_KEY ao .env"
        )
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


def get_llm(tier: str = "heavy"):
    """
    Provider-agnostic LLM factory.
    Provider resolution: _thread_local.provider (benchmark) -> LLM_PROVIDER env var
    """
    current_provider = (
        getattr(_thread_local, "provider", None)
        or os.environ.get("LLM_PROVIDER", LLM_PROVIDER)
    )

    if current_provider in AZURE_MODELS:
        return ChatOpenAI(
            model=AZURE_MODELS[current_provider][tier],
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
        )

    if current_provider == "gemini-lite":
        return _make_gemini(GEMINI_FALLBACK)

    if current_provider == "gemini-pro":
        return _make_gemini("gemini-2.5-pro")

    return _make_gemini(GEMINI_PRIMARY)
