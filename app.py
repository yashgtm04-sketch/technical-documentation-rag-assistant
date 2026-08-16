"""
Phase 14 — Streamlit Deployment.

This is the only file that imports `streamlit`. Every actual RAG operation
(loading, chunking, embedding, retrieval, prompting, generation) lives in
src/ and is imported here — the UI layer's only job is to collect input,
call the pipeline, and render output. This separation is what makes each
phase independently testable from the command line (see the `if __name__ ==
"__main__"` block in every src/ module) without ever starting a Streamlit
server.
"""

import shutil

import streamlit as st
from dotenv import load_dotenv

# Must run before any `src` import: src/config.py reads HF_API_TOKEN from the
# environment at import time, so .env has to be loaded first or the token is
# always read as empty — even when the file has a real value in it.
# On Streamlit Community Cloud, set HF_API_TOKEN under app Settings -> Secrets
# instead — .env files are not part of the deployed repo, and load_dotenv()
# is a harmless no-op there since no .env file exists.
load_dotenv()

from src.config import DATA_DIR, INDEX_DIR
from src.document_loader import load_pdfs
from src.preprocessing import preprocess_documents
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store, save_vector_store, load_vector_store, index_exists
from src.pipeline import answer_question

st.set_page_config(page_title="Technical Documentation Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedding_model_cached():
    """Loading the embedding model is expensive (~seconds); cache it across reruns.

    Streamlit reruns the whole script on every interaction (Phase 14 data
    flow), so without caching we would reload this model on every button
    click and every widget change.
    """
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def get_vector_store_cached(_embedding_model, index_signature: str):
    """Load the FAISS index from disk. `index_signature` is a cache-busting
    key (see below) — when it changes, Streamlit knows to reload instead of
    reusing a stale cached index."""
    return load_vector_store(INDEX_DIR, _embedding_model)


def rebuild_index_from_data_dir(embedding_model):
    with st.spinner("Loading PDFs, chunking, and embedding — this runs once and is then cached..."):
        raw_docs = load_pdfs(DATA_DIR)
        clean_docs = preprocess_documents(raw_docs)
        chunks = chunk_documents(clean_docs)
        store = build_vector_store(chunks, embedding_model)
        save_vector_store(store, INDEX_DIR)
    return store


def main():
    st.title("📚 Technical Documentation Assistant")
    st.caption(
        "Retrieval-Augmented Generation over Python, Pandas, NumPy, Scikit-learn, "
        "LangChain, FAISS, and Streamlit documentation."
    )

    with st.sidebar:
        st.header("Knowledge Base")
        st.write(f"Documents folder: `{DATA_DIR.name}/`")

        existing_pdfs = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
        if existing_pdfs:
            st.write("Loaded documentation:")
            for name in existing_pdfs:
                st.write(f"- {name}")
        else:
            st.warning("No PDFs found in data/. Upload one below.")

        uploaded_files = st.file_uploader(
            "Upload additional documentation (PDF)", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                dest = DATA_DIR / uploaded_file.name
                with open(dest, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
            st.success(f"Saved {len(uploaded_files)} file(s) to data/. Click 'Rebuild index' below.")

        rebuild_clicked = st.button("🔄 Rebuild index from data/", use_container_width=True)

        st.divider()
        st.header("How this works")
        st.markdown(
            "1. PDFs are loaded and cleaned\n"
            "2. Text is split into overlapping chunks\n"
            "3. Chunks are embedded with a Sentence Transformer\n"
            "4. Embeddings are stored in a FAISS index\n"
            "5. Your question is embedded and matched against the index\n"
            "6. Only the matched chunks are sent to the LLM as context\n"
            "7. If nothing matches well enough, the app says so instead of guessing"
        )

    embedding_model = get_embedding_model_cached()

    if rebuild_clicked:
        get_vector_store_cached.clear()
        vector_store = rebuild_index_from_data_dir(embedding_model)
    elif index_exists(INDEX_DIR):
        # index_signature busts the Streamlit cache whenever the on-disk
        # index file changes size (i.e. after a rebuild), without needing a
        # manual cache-clear call on every code path.
        index_signature = str((INDEX_DIR / "index.faiss").stat().st_mtime)
        vector_store = get_vector_store_cached(embedding_model, index_signature)
    else:
        st.info("No saved index found yet. Building one from data/ now...")
        vector_store = rebuild_index_from_data_dir(embedding_model)

    st.divider()
    question = st.text_input(
        "Ask a technical question",
        placeholder="e.g. How do I merge two DataFrames in pandas?",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Retrieving relevant documentation and generating an answer..."):
            result = answer_question(question, vector_store)

        if result.grounded:
            st.success("Answer")
        else:
            st.warning("Answer (not fully grounded — see note)")
        st.write(result.answer)

        if result.sources:
            st.subheader("Retrieved Source Passages")
            for i, src in enumerate(result.sources, start=1):
                with st.expander(f"[{i}] {src.source} — page {src.page} (similarity: {src.score:.2f})"):
                    st.text(src.text)
    elif ask_clicked:
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
