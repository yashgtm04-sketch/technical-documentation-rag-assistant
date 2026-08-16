"""
Phase 5 — Embedding Generation.

Problem:
    To retrieve chunks by *meaning* rather than exact keyword overlap, we
    need to represent each chunk (and later, each query) as a vector such
    that semantically similar text ends up close together in vector space.

Why a Hugging Face Sentence Transformer (all-MiniLM-L6-v2):
    It is specifically trained (via contrastive learning on sentence pairs)
    so that cosine similarity between embeddings tracks semantic similarity.
    It is small (~80MB, 384-dim output), runs fast on CPU, and is one of the
    most widely used embedding models in the LangChain/CampusX ecosystem —
    easy to justify and explain in an interview.

Why not keyword-based retrieval (TF-IDF / BM25) instead of embeddings:
    Keyword matching only finds chunks that share literal tokens with the
    query. If a user asks "How do I combine two tables in pandas?" but the
    documentation says "merge DataFrames", a keyword search finds nothing
    useful because there's no token overlap. Embeddings capture that
    "combine two tables" and "merge DataFrames" mean the same thing, because
    the model was trained on meaning, not exact words. This is the central
    justification for using embeddings at all in a RAG system.

Why not a larger embedding model (e.g. an OpenAI or a 1B+ parameter model):
    all-MiniLM-L6-v2 is free, runs locally with no API key or GPU, and is
    accurate enough for short technical documentation chunks. A larger model
    would improve retrieval marginally at a real cost in latency and infra
    complexity — not a good tradeoff for this project's scope.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a LangChain-compatible embedding model wrapper.

    Using LangChain's wrapper (rather than calling sentence-transformers
    directly) means this object can be passed straight into FAISS.from_documents
    in the next phase with no extra glue code.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},  # required for cosine similarity search
    )


if __name__ == "__main__":
    model = get_embedding_model()
    sample_vector = model.embed_query("How do I merge two DataFrames in pandas?")
    print(f"Embedding dimension: {len(sample_vector)}")
    print(f"First 5 values: {sample_vector[:5]}")
