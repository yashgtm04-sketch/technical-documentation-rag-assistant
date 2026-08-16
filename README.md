# Technical Documentation Assistant (RAG + LLM)

A Retrieval-Augmented Generation (RAG) assistant that answers programming and machine learning questions using only official documentation — Python, Pandas, NumPy, Scikit-learn, LangChain, FAISS, and Streamlit — instead of an LLM's own (potentially outdated or hallucinated) knowledge.

## Problem it solves

Developers switch between multiple documentation sites while coding, and LLMs asked the same questions directly can produce confident-sounding but unsupported answers because they rely on pretrained knowledge rather than the actual docs. This project retrieves the relevant passage first and forces the LLM to answer only from it, with the source passage shown alongside every answer.

## Architecture

```
Documentation PDFs (Python, Pandas, NumPy, Scikit-learn, LangChain, FAISS, Streamlit)
        │
        ▼
  PyPDFLoader (load + extract text per page)
        │
        ▼
  Text Cleaning (remove PDF artifacts, keep headings/code)
        │
        ▼
  RecursiveCharacterTextSplitter (overlapping chunks)
        │
        ▼
  Sentence-Transformer Embeddings (all-MiniLM-L6-v2)
        │
        ▼
  FAISS Vector Store (persisted to disk)
────────────────────────────────────────────
  User Question
        │
        ▼
  Query Embedding (same model as above)
        │
        ▼
  FAISS Similarity Search (Top-K)
        │
        ▼
  Confidence Guardrail (score below threshold → "not found", skip LLM)
        │
        ▼
  Prompt Construction (context + question + "answer only from context")
        │
        ▼
  Mistral-7B-Instruct-v0.2 (via Hugging Face Inference Providers)
        │
        ▼
  Answer + Source Attribution (document name, page, retrieved passage)
        │
        ▼
  Streamlit UI
```

## Why these design choices

- **FAISS over linear search**: FAISS is a purpose-built vector similarity search library; even its simplest index is a vectorized, optimized version of brute-force comparison, and it gives index persistence (save/load) for free.
- **Sentence-Transformer embeddings over keyword search**: embeddings capture meaning ("combine two tables" ≈ "merge DataFrames"), while keyword/TF-IDF search only matches literal tokens.
- **Overlapping chunks**: prevents context (like a code example) from being split across two chunks with neither half being individually useful.
- **Confidence guardrail**: if the best-matching chunk's similarity score is below a threshold, the app returns "not found in the documentation" instead of letting the LLM guess. This is the project's main defense against hallucination.
- **Hugging Face Inference Providers instead of local GPU hosting**: runs an instruction-tuned model (Mistral-7B-Instruct-v0.2, pinned in `src/config.py`) without requiring a local GPU or a ~15GB model download, keeping the project's complexity in the RAG logic rather than in ML infrastructure. Note: not every model on the Hub is actually servable this way — a model must be deployed by at least one Inference Provider, which you can check on its Hugging Face model page. Mistral-7B-Instruct-v0.3 (a version newer than the one pinned here) is *not* currently deployed by any provider, which is why v0.2 is used instead.

See the docstring at the top of each file in `src/` for the full problem/solution/alternatives reasoning behind that specific phase — every module is written to be read and defended independently.

## Project structure

```
tech_doc_assistant/
├── data/                     # Sample documentation PDFs (7 libraries)
├── src/
│   ├── config.py             # All tunable constants in one place
│   ├── document_loader.py    # Phase 2 — PDF loading
│   ├── preprocessing.py      # Phase 3 — text cleaning
│   ├── chunking.py           # Phase 4 — recursive chunking
│   ├── embeddings.py         # Phase 5 — Sentence Transformer embeddings
│   ├── vector_store.py       # Phase 6 — FAISS build/save/load
│   ├── retrieval.py          # Phases 7-9 — query embedding, search, guardrail
│   ├── prompt_builder.py     # Phase 10 — grounded prompt construction
│   ├── llm.py                # Phase 11 — Mistral-7B-Instruct-v0.2 via HF Inference Providers
│   └── pipeline.py           # Phase 12 — orchestration + source attribution
├── build_index.py            # One-time indexing script (Phases 1-6)
├── app.py                    # Phase 14 — Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

Every module in `src/` has an `if __name__ == "__main__":` block so it can be run and inspected on its own, e.g. `python src/chunking.py`, before wiring it into the full pipeline.

## Setup

1. **Clone and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get a Hugging Face API token**

   Create a **fine-grained** token at https://huggingface.co/settings/tokens/new with the
   "Make calls to Inference Providers" permission enabled, then:

   ```bash
   cp .env.example .env
   # edit .env and paste your token into HF_API_TOKEN
   ```

   **On cost**: every Hugging Face account (including free accounts) gets a small monthly
   credit allocation for Inference Providers — enough for testing and demoing this project,
   but not for heavy sustained use. Once free-tier credits run out, requests simply stop
   until they reset the next month (a paid PRO plan raises the limit). Check current limits
   at https://huggingface.co/docs/inference-providers/pricing before a demo.

3. **Build the FAISS index** (one-time; re-run whenever you change the PDFs in `data/`)

   ```bash
   python build_index.py
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this whole `tech_doc_assistant/` folder to a GitHub repo (as the repo root, not nested inside another folder).
2. On https://share.streamlit.io, create a new app pointing at that repo with **main file path set to `app.py`**.
3. `.env` files are not deployed with your repo, so `HF_API_TOKEN` won't be picked up from it on Cloud. Instead, go to your app's **Settings → Secrets** and add:
   ```
   HF_API_TOKEN = "your_token_here"
   ```
4. Streamlit Cloud installs whatever is listed in `requirements.txt` at the repo root — make sure you didn't rename or move that file.

If you see `ModuleNotFoundError` on deploy, it almost always means either `src/__init__.py` is missing from the repo, or a file inside `src/` imports a sibling module by a bare name (e.g. `from retrieval import ...`) instead of `from src.retrieval import ...`. Every file in this project already uses the package-qualified form for exactly this reason — if you edit these files, keep that pattern.

## Sample dataset

`data/` ships with one official documentation excerpt per library (Python data structures, pandas DataFrames, NumPy array creation, scikit-learn getting started, LangChain retrieval concepts, FAISS getting started, and Streamlit basic concepts), sourced directly from each project's official docs. This keeps the project runnable out of the box. To scale it up, add more official PDFs to `data/` and re-run `build_index.py` — nothing else needs to change.

## Notes

- **LLM availability depends on a third-party provider.** `HF_LLM_MODEL` in `config.py` must be a model actually deployed by a Hugging Face Inference Provider — this can change over time as providers add or drop models. If `llm.py` starts returning errors, check the model's "Inference Providers" panel on its Hugging Face page and swap in a currently-deployed alternative.
- **Chunking is not code-aware.** A single splitter is used for both prose and code, so a code block could in principle be split across two chunks (mitigated by the chunk overlap).
- **No conversation memory.** Each question is answered independently; a follow-up question won't resolve context from an earlier turn.
