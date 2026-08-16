"""
Phase 2 — Document Loading.

Problem:
    We have PDF documentation files sitting on disk. Before we can search
    them semantically, we need their raw text pulled out and tagged with
    where it came from.

Why PyPDFLoader:
    LangChain's PyPDFLoader gives us one Document object per PDF page, and
    automatically attaches metadata (source file path, page number) to each
    one. That metadata is what lets us show "Source: pandas_docs.pdf, page 3"
    in the UI later (Phase 11) instead of just a wall of text.

Why not a raw PDF library (pypdf/pdfplumber) directly:
    We could call pypdf ourselves and build Document objects by hand, but
    PyPDFLoader already does that correctly and keeps us inside the
    LangChain ecosystem, so the output plugs straight into the text splitter
    in the next phase without any glue code.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdfs(data_dir: Path) -> List[Document]:
    """Load every PDF in `data_dir` and return a flat list of page-level Documents.

    Each Document's metadata already contains:
        - 'source': the PDF file path
        - 'page':   the zero-indexed page number

    Files that fail to load (corrupted PDF, wrong permissions, etc.) are
    skipped with a warning rather than crashing the whole pipeline — one bad
    file should not block indexing the other six.
    """
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Add documentation PDFs before building the index."
        )

    all_documents: List[Document] = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            # Normalize the source name to just the filename — nicer for
            # display than a full absolute path in the Streamlit UI.
            for page in pages:
                page.metadata["source"] = pdf_path.name
            all_documents.extend(pages)
            logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
        except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
            logger.warning("Skipping %s — failed to load (%s)", pdf_path.name, exc)

    if not all_documents:
        raise RuntimeError("All PDF files failed to load. Check the files in the data/ folder.")

    return all_documents


if __name__ == "__main__":
    # Quick standalone test: `python -m src.document_loader` (from the project root)
    from src.config import DATA_DIR

    logging.basicConfig(level=logging.INFO)
    docs = load_pdfs(DATA_DIR)
    print(f"Total pages loaded: {len(docs)}")
    print(f"Example metadata: {docs[0].metadata}")
    print(f"Example text snippet: {docs[0].page_content[:200]}")
