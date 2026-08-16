"""
Phase 3 — Text Preprocessing.

Problem:
    Text extracted from PDFs is messy: repeated headers/footers, broken line
    breaks in the middle of sentences, stray page numbers, and extra
    whitespace. If we embed and index this noise as-is, it dilutes the
    semantic signal in every chunk.

Why simple regex/string cleanup (and not a heavier NLP pipeline):
    We are not trying to "understand" the text at this stage, only to remove
    formatting artifacts while preserving things that carry meaning —
    headings and code blocks. A full NLP cleaning pipeline (stemming,
    stopword removal, lowercasing) would actually hurt retrieval quality
    here, because embedding models are trained on natural, cased text and
    code syntax is case- and symbol-sensitive (`DataFrame` != `dataframe`).
"""

import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Remove common PDF-extraction artifacts while preserving structure."""
    # Collapse 3+ blank lines into a single blank line (keeps paragraph breaks).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix words split across a line break by a hyphen, e.g. "data-\nframe" -> "dataframe".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join a line break that occurs mid-sentence (lowercase letter continues
    # on the next line) into a single line, but keep breaks after punctuation
    # or before a new heading/bullet.
    text = re.sub(r"(?<![.:\n])\n(?!\n|[-#>*\d])", " ", text)

    # Collapse repeated spaces/tabs introduced by column layouts in PDFs.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip trailing/leading whitespace per line.
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def preprocess_documents(documents: list[Document]) -> list[Document]:
    """Apply `clean_text` to every Document, keeping metadata untouched."""
    cleaned = []
    for doc in documents:
        cleaned_text = clean_text(doc.page_content)
        if cleaned_text:  # drop pages that are empty after cleaning (e.g. blank cover pages)
            cleaned.append(Document(page_content=cleaned_text, metadata=doc.metadata))
    return cleaned


if __name__ == "__main__":
    # Run as: python -m src.preprocessing  (from the project root)
    from src.config import DATA_DIR
    from src.document_loader import load_pdfs

    raw_docs = load_pdfs(DATA_DIR)
    clean_docs = preprocess_documents(raw_docs)
    print(f"Pages before cleaning: {len(raw_docs)} | after cleaning: {len(clean_docs)}")
    print("--- Before ---")
    print(raw_docs[0].page_content[:300])
    print("--- After ---")
    print(clean_docs[0].page_content[:300])
