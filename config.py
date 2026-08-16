"""
Central configuration for the Technical Documentation Assistant.

Keeping every tunable value in one place means every other module imports
from here instead of hardcoding constants. This is a small thing, but it is
exactly the kind of "production hygiene" detail interviewers look for.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vector_index"          # FAISS index is persisted here
EVAL_DIR = BASE_DIR / "eval"

# --- Phase 4: Chunking ---------------------------------------------------
CHUNK_SIZE = 700          # characters per chunk
CHUNK_OVERLAP = 120       # overlap between consecutive chunks

# --- Phase 5: Embeddings -------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Phase 8: Retrieval ---------------------------------------------------
TOP_K = 4                          # how many chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.15        # guardrail cutoff, see retrieval.py

# --- Phase 11: LLM ---------------------------------------------------------
# Hugging Face Inference Providers (the router API that replaced the old
# serverless "Inference API") is used instead of local GPU hosting, so the
# project runs on a laptop without a GPU.
#
# IMPORTANT: not every model on the Hub is actually servable through this
# API — a model must be deployed by at least one Inference Provider (check
# the "Inference Providers" panel on the model's Hugging Face page; if it
# says "This model isn't deployed by any Inference Provider", pick a
# different one). Mistral-7B-Instruct-v0.3 is NOT currently deployed by any
# provider, so this project pins v0.2 instead, which is confirmed live via
# the Featherless AI provider and is fully open (Apache-2.0, no gated
# access request needed) — an important distinction from Llama-3-Instruct,
# which requires accepting Meta's license on the Hub before it will work.
#
# The ":featherless-ai" suffix pins the provider explicitly. Without it, the
# router's auto-provider-selection can return
# "The requested model ... is not supported by any provider you have
# enabled" (400) for accounts that haven't explicitly enabled a provider,
# even though the model is genuinely deployed there — confirmed by testing
# directly against Featherless AI.
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2   # low temperature: we want grounded, repeatable answers
