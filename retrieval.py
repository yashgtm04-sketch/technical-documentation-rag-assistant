"""
Phases 7, 8, 9 — Query Processing, Semantic Retrieval, and the Retrieval Guardrail.

Problem:
    Given a raw user question, we need to (a) embed it with the exact same
    model used for the chunks, (b) find the Top-K nearest chunks in FAISS,
    and (c) decide whether those chunks are actually relevant enough to
    answer from — otherwise we risk the LLM "filling gaps" with its own
    pretrained knowledge, which is the hallucination failure mode this whole
    project exists to prevent.

How semantic retrieval works (for the interview-facing explanation):
    The query is embedded into the same vector space as the document chunks
    using the same Sentence Transformer model. FAISS then computes the
    similarity (here, via normalized embeddings + inner product, which is
    equivalent to cosine similarity) between the query vector and every
    indexed chunk vector, and returns the K chunks with the highest score.
    Because embeddings encode meaning rather than exact wording, this finds
    relevant chunks even when the user's phrasing differs from the
    documentation's phrasing.

Why a similarity threshold guardrail (Phase 9):
    Retrieval always returns the K best chunks available, even if none of
    them are actually relevant to the query (e.g. asking about Docker in a
    knowledge base that only has Python/Pandas/NumPy docs). Without a
    cutoff, the LLM would be handed irrelevant context and could still
    generate a plausible-sounding but ungrounded answer. Checking the top
    similarity score against a fixed threshold lets the system say "not
    found in the documentation" instead of guessing — this is the single
    biggest lever this project has against hallucination, and it costs one
    `if` statement.
"""

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import TOP_K, SIMILARITY_THRESHOLD


@dataclass
class RetrievalResult:
    chunks: list[Document]
    scores: list[float]
    is_confident: bool  # False => guardrail tripped, caller should not generate an answer


def retrieve(query: str, vector_store: FAISS, k: int = TOP_K) -> RetrievalResult:
    """Embed `query`, run FAISS similarity search, and apply the confidence guardrail.

    FAISS's `similarity_search_with_relevance_scores` returns scores already
    normalized to roughly [0, 1] (higher = more similar), which is what the
    threshold in config.py is calibrated against.
    """
    results_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=k)

    if not results_with_scores:
        return RetrievalResult(chunks=[], scores=[], is_confident=False)

    chunks = [doc for doc, _score in results_with_scores]
    scores = [score for _doc, score in results_with_scores]

    is_confident = scores[0] >= SIMILARITY_THRESHOLD
    return RetrievalResult(chunks=chunks, scores=scores, is_confident=is_confident)


if __name__ == "__main__":
    # Run as: python -m src.retrieval  (from the project root)
    from src.config import INDEX_DIR
    from src.embeddings import get_embedding_model
    from src.vector_store import load_vector_store

    model = get_embedding_model()
    store = load_vector_store(INDEX_DIR, model)

    for question in [
        "How do I merge two DataFrames in pandas?",
        "What is the airspeed velocity of an unladen swallow?",  # should trip the guardrail
    ]:
        result = retrieve(question, store)
        print(f"\nQ: {question}")
        print(f"Top score: {result.scores[0] if result.scores else 'N/A'} | Confident: {result.is_confident}")
        for chunk, score in zip(result.chunks, result.scores):
            print(f"  [{score:.3f}] {chunk.metadata['source']} -> {chunk.page_content[:80]}")
