import itertools
import os
import threading

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Azure model names — must match deployment names in Foundry exactly
AZURE_MODELS = {
    "azure":      {"heavy": "gpt-4o",                     "cheap": "gpt-4.1"},
    "azure-41":   {"heavy": "gpt-4.1",                    "cheap": "gpt-4.1"},
    "azure-kimi": {"heavy": "Kimi-K2.5",                  "cheap": "Kimi-K2.5"},
    "azure-grok": {"heavy": "grok-4-1-fast-non-reasoning","cheap": "grok-4-1-fast-non-reasoning"},
    "azure-nano": {"heavy": "gpt-5.4-nano",               "cheap": "gpt-5.4-nano"},
}

# Gemini model cascade (most capable → most available)
# gemini-2.5-flash:      GA, best for heavy nodes (default)
# gemini-2.5-pro:        thinking model, slow — only for gemini-pro provider
# gemini-2.0-flash-lite: GA, fastest/cheapest, fallback
# gemini-2.0-flash:      SHUT DOWN — do not use
GEMINI_PRIMARY  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK = "gemini-2.0-flash-lite"

# ── Thread-local provider override (used by benchmark tab) ───────────────────
# Each ThreadPoolExecutor worker sets _thread_local.provider before calling
# get_llm(), avoiding the os.environ race condition between parallel threads.
_thread_local = threading.local()

# ── Lazy Gemini key rotation ──────────────────────────────────────────────────
# Initialized on first _get_gemini_key() call so load_dotenv() has already run.
# Thread-safe via _key_lock.
_key_lock = threading.Lock()
_key_cycle = None


def _get_gemini_key() -> str:
    """Return next Gemini API key in round-robin rotation. Thread-safe + lazy."""
    global _key_cycle
    with _key_lock:
        if _key_cycle is None:
            keys = [k for k in [
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("GOOGLE_API_KEY_2"),
            ] if k]
            if not keys:
                raise ValueError(
                    "Nenhuma GOOGLE_API_KEY configurada no .env. "
                    "Adicione GOOGLE_API_KEY ou GOOGLE_API_KEY_2."
                )
            _key_cycle = itertools.cycle(keys)
        return next(_key_cycle)


def _make_gemini(model: str) -> ChatGoogleGenerativeAI:
    """Instantiate a Gemini model with the next available API key."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_get_gemini_key(),
        temperature=0,
    )


def is_quota_error(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()


def get_llm(tier: str = "heavy"):
    """
    Provider-agnostic LLM factory.

    Provider resolution order:
      1. _thread_local.provider  — set per-thread by benchmark (no race condition)
      2. LLM_PROVIDER env var    — set by Tab 1 sidebar or .env default

    Providers:
      "gemini"      → gemini-3.5-flash → gemini-2.0-flash-lite (quota fallback)
      "gemini-pro"  → gemini-2.5-pro → gemini-3.5-flash → gemini-2.0-flash-lite
      "azure"       → GPT-4o (heavy) / GPT-4.1 (cheap)
      "azure-41"    → GPT-4.1 for all tiers
      "azure-kimi"  → Kimi K2.5 for all tiers
      "azure-grok"  → Grok 4.1 Fast for all tiers
      "azure-nano"  → GPT-5.4 Nano for all tiers
    """
    current_provider = (
        getattr(_thread_local, "provider", None)
        or os.environ.get("LLM_PROVIDER", LLM_PROVIDER)
    )

    # ── Azure AI Foundry — same endpoint + key for all models ────────────────
    if current_provider in AZURE_MODELS:
        model = AZURE_MODELS[current_provider][tier]
        return ChatOpenAI(
            model=model,
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
        )

    # ── Gemini Lite: 3.1-flash-lite direct (no fallback — it IS the fallback) ─
    if current_provider == "gemini-lite":
        return _make_gemini(GEMINI_FALLBACK)

    # ── Gemini Pro tier: 2.5-pro → 3.5-flash → 3.1-flash-lite ─────────────
    if current_provider == "gemini-pro":
        primary  = _make_gemini("gemini-2.5-pro")
        mid      = _make_gemini(GEMINI_PRIMARY)
        fallback = _make_gemini(GEMINI_FALLBACK)
        return primary.with_fallbacks([mid, fallback])

    # ── Gemini default: 3.5-flash → 3.1-flash-lite ──────────────────────────
    primary  = _make_gemini(GEMINI_PRIMARY)
    fallback = _make_gemini(GEMINI_FALLBACK)
    return primary.with_fallbacks([fallback])
