"""
One-time indexing script — runs Phases 1 through 6.

Run this once (and again any time you add/remove PDFs in data/):

    python build_index.py

It loads every PDF in data/, cleans and chunks the text, embeds every chunk,
and saves the resulting FAISS index to vector_index/. The Streamlit app then
just loads this saved index at startup instead of repeating this work on
every run — see src/vector_store.py for why persistence matters.
"""

import logging

from src.config import DATA_DIR, INDEX_DIR
from src.document_loader import load_pdfs
from src.preprocessing import preprocess_documents
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store, save_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Phase 2: Loading PDFs from %s", DATA_DIR)
    raw_documents = load_pdfs(DATA_DIR)

    logger.info("Phase 3: Cleaning text")
    clean_documents = preprocess_documents(raw_documents)

    logger.info("Phase 4: Chunking documents")
    chunks = chunk_documents(clean_documents)
    logger.info("Created %d chunks from %d pages", len(chunks), len(clean_documents))

    logger.info("Phase 5: Loading embedding model")
    embedding_model = get_embedding_model()

    logger.info("Phase 6: Building FAISS index")
    vector_store = build_vector_store(chunks, embedding_model)
    save_vector_store(vector_store, INDEX_DIR)

    logger.info("Done. Index saved to %s", INDEX_DIR)
    logger.info("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
