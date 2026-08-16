"""
Phase 4 — Document Chunking.

Problem:
    A single PDF page can be several hundred to a few thousand characters.
    Embedding an entire page as one vector blurs together multiple ideas
    (e.g. "list methods" and "tuples" on the same page), which hurts
    retrieval precision — the embedding becomes an average of everything on
    the page instead of a focused representation of one concept.

Why RecursiveCharacterTextSplitter:
    It tries to split on natural boundaries first (paragraph breaks, then
    sentences, then words) before falling back to a hard character cut.
    This keeps related sentences together far more often than a naive
    fixed-length slice would, without requiring any language-specific NLP.

Why overlapping chunks:
    Important context often straddles a chunk boundary — e.g. a code
    example that starts in chunk N and finishes in chunk N+1. A 100-150
    character overlap means both chunks retain enough surrounding context
    to be individually meaningful, at the cost of a small amount of
    duplicated content in the index. Without overlap, retrieval can return
    a chunk that references "the example above" with no example actually
    present in it.

Why not chunk by fixed token count only, or split code blocks separately:
    A fixed-size splitter with no separator hierarchy would cut mid-word or
    mid-sentence unpredictably. Splitting code blocks into their own
    chunking pipeline is a legitimate improvement (see README's future
    enhancements) but adds a second code path; for this project's scope, one
    consistent splitter for all content keeps the pipeline easy to explain
    and maintain end-to-end.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split cleaned page-level Documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tried in order, most natural first
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a stable, human-readable id: useful for debugging and
    # for the eval script to reference "which chunk answered this question".
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


if __name__ == "__main__":
    # Run as: python -m src.chunking  (from the project root)
    from src.config import DATA_DIR
    from src.document_loader import load_pdfs
    from src.preprocessing import preprocess_documents

    raw_docs = load_pdfs(DATA_DIR)
    clean_docs = preprocess_documents(raw_docs)
    chunks = chunk_documents(clean_docs)

    print(f"Pages: {len(clean_docs)} -> Chunks: {len(chunks)}")
    print(f"Average chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print("--- Sample chunk ---")
    print(chunks[0].metadata)
    print(chunks[0].page_content)
