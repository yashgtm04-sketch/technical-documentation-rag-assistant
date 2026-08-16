"""
Phase 11 — Response Generation.

Problem:
    We need an instruction-tuned LLM to turn (context + question) into a
    fluent, grounded answer. Running a 7B-parameter model locally requires a
    GPU with several GB of VRAM (or a slow, heavily quantized CPU setup),
    which is impractical for a student laptop and adds a lot of setup
    complexity that has nothing to do with RAG itself.

Why Hugging Face Inference Providers (hosted Mistral-7B-Instruct):
    It gives us an instruction-tuned model called with a single HTTP
    request, with no local GPU, no model download (~15GB), and no
    quantization tooling. This keeps the project's complexity concentrated
    in the RAG logic (retrieval, chunking, guardrails) rather than in ML
    infrastructure — which is also the part that is actually being
    evaluated here.

Why the /v1/chat/completions router endpoint specifically:
    Hugging Face replaced the old serverless "Inference API"
    (api-inference.huggingface.co/models/{model}, which used a raw
    {"inputs": ...} payload) with "Inference Providers": a unified,
    OpenAI-compatible router (router.huggingface.co/v1/chat/completions)
    that forwards the request to whichever backend (Featherless AI, Groq,
    Together, etc.) actually hosts the requested model. Not every model on
    the Hub is hosted by a provider — this is why HF_LLM_MODEL in config.py
    is pinned to a model confirmed to be live (Mistral-7B-Instruct-v0.2 via
    Featherless AI) rather than a newer version that looked equivalent but
    isn't actually deployed anywhere.

Why not a local llama-cpp-python / GGUF quantized model:
    That is a legitimate production alternative (mentioned in the README's
    future enhancements) but it introduces a second, unrelated skill set
    (model quantization, GGUF formats, llama.cpp build tooling) that would
    make this project harder to explain end-to-end without adding anything
    to the RAG pipeline's actual logic.

A note on cost: Hugging Face gives every account a small monthly free
credit allocation for Inference Providers, enough for light testing/demo
use. Once that's used up, free accounts simply get blocked until next
month (no pay-as-you-go without a PRO subscription) — see the README for
details on the account tiers.
"""

import logging

import requests

from src.config import HF_LLM_MODEL, HF_API_TOKEN, MAX_NEW_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


class LLMGenerationError(Exception):
    """Raised when the Hugging Face Inference Providers call fails or times out."""


def generate_answer(prompt: str) -> str:
    """Send `prompt` to the hosted instruction-tuned LLM and return its text response.

    Uses the OpenAI-compatible chat completions format that Inference
    Providers expects: a list of role/content messages rather than the
    older raw-text `inputs` payload. Our full grounded prompt (system
    instructions + context + question, all built in prompt_builder.py) is
    sent as a single user message, which keeps this function simple while
    still giving the model everything it needs.

    Errors (missing API token, rate limiting, network timeout, a model that
    isn't actually deployed by any provider) are caught and re-raised as a
    single LLMGenerationError so the Streamlit layer can show one clean
    error message instead of a raw traceback.
    """
    if not HF_API_TOKEN:
        raise LLMGenerationError(
            "HF_API_TOKEN is not set. Add it to your .env file (see .env.example)."
        )

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "model": HF_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    try:
        response = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise LLMGenerationError("The LLM API request timed out. Please try again.") from exc
    except requests.exceptions.HTTPError as exc:
        # The router's JSON body (e.g. "model not supported by any provider
        # you have enabled", "model is busy") is far more diagnostic than the
        # bare status code — surface it instead of hiding it behind a generic
        # 404/model-error hint that may not even be the actual cause.
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            detail = response.text
        raise LLMGenerationError(f"LLM API returned an error: {exc}. Details: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise LLMGenerationError(f"Could not reach the LLM API: {exc}") from exc

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMGenerationError(f"Unexpected response format from LLM API: {data}") from exc


if __name__ == "__main__":
    # Run as: python -m src.llm  (from the project root)
    from src.prompt_builder import build_prompt
    from langchain_core.documents import Document

    logging.basicConfig(level=logging.INFO)
    chunk = Document(
        page_content="df.merge(other_df, on='key') combines two DataFrames on a shared column.",
        metadata={"source": "pandas_docs.pdf", "page": 12},
    )
    test_prompt = build_prompt("How do I merge two DataFrames?", [chunk])
    print(generate_answer(test_prompt))
