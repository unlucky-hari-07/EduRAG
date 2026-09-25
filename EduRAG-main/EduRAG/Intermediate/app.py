"""
app.py

EduRAG — Document-Based Q&A Assistant (Intermediate Level)

Full RAG pipeline, orchestrated here:

    Document Upload
        |
    File Validation           (utils/validators.py)
        |
    Text Extraction           (rag/loaders.py)
        |
    Text Cleaning             (rag/cleaner.py)
        |
    Chunking                  (rag/chunker.py)
        |
    Embedding Generation      (rag/embeddings.py)
        |
    FAISS Vector Index        (rag/vector_store.py)
        |
    User Question
        |
    Question Embedding        (rag/embeddings.py)
        |
    Similarity Search         (rag/retriever.py)
        |
    Top-K Relevant Chunks
        |
    Context Construction      (rag/retriever.py)
        |
    Ollama                     (rag/generator.py)
        |
    Grounded Answer + Sources

Run with:
    streamlit run app.py
"""

import uuid

import streamlit as st

from rag.chunker import chunk_pages
from rag.embeddings import embed_texts
from rag.generator import generate_answer
from rag.loaders import DocumentLoadError, load_document
from rag.retriever import build_context, retrieve_chunks
from rag.vector_store import VectorStore, VectorStoreError
from utils import config
from utils.validators import validate_extracted_text, validate_question, validate_upload

st.set_page_config(page_title="EduRAG Document Q&A", page_icon="📚", layout="wide")


def get_store() -> VectorStore:
    """Return this session's VectorStore, creating it on first use."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(dimension=config.EMBEDDING_DIMENSION)
    return st.session_state.vector_store


def render_api_key_warning() -> None:
    if not config.is_api_key_configured():
        st.warning(
            "⚠️ Could not reach Ollama. Make sure it's installed and running "
            "(`ollama serve`), and that `OLLAMA_BASE_URL`/`OLLAMA_MODEL_NAME` "
            "in your `.env` file are set correctly. See `.env.example` for "
            "the expected format."
        )


def process_upload(uploaded_file) -> None:
    """Run the full ingestion pipeline on a Streamlit-uploaded file."""
    raw_bytes = uploaded_file.getvalue()

    is_valid, message = validate_upload(uploaded_file.name, len(raw_bytes))
    if not is_valid:
        st.error(message)
        return

    import tempfile
    from pathlib import Path

    with st.status("Processing document...", expanded=True) as status:
        try:
            status.write("📄 Extracting text...")
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = Path(tmp.name)

            try:
                pages = load_document(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            combined_text = "\n".join(text for _, text in pages)
            is_valid, message = validate_extracted_text(combined_text)
            if not is_valid:
                status.update(label="Extraction failed", state="error")
                st.error(message)
                return

            status.write("🧹 Cleaning and chunking text...")
            chunks = chunk_pages(
                pages,
                chunk_size=config.CHUNK_SIZE_WORDS,
                overlap=config.CHUNK_OVERLAP_WORDS,
            )

            if not chunks:
                status.update(label="Chunking failed", state="error")
                st.error("No usable text remained after cleaning and chunking this document.")
                return

            status.write(f"🔢 Generating embeddings for {len(chunks)} chunks...")
            texts = [c.text for c in chunks]
            vectors = embed_texts(texts)

            status.write("📥 Indexing in FAISS...")
            document_id = str(uuid.uuid4())
            store = get_store()
            store.add_document(document_id, uploaded_file.name, chunks, vectors)

            status.update(label="Document processed successfully!", state="complete")

        except DocumentLoadError as exc:
            status.update(label="Extraction failed", state="error")
            st.error(str(exc))
            return
        except VectorStoreError as exc:
            status.update(label="Indexing failed", state="error")
            st.error(f"Failed to index document: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - never leak a raw stack trace to the user
            status.update(label="Processing failed", state="error")
            st.error(
                "An unexpected error occurred while processing this document. "
                "Please try again."
            )
            return

    st.success(f"'{uploaded_file.name}' indexed with {len(chunks)} chunks.")


def render_document_status() -> None:
    store = get_store()
    documents = store.list_documents()

    st.subheader("📂 Indexed Documents")

    if not documents:
        st.info("No documents indexed yet. Upload one above to get started.")
        return

    for doc in documents:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{doc['document_name']}** — {doc['chunk_count']} chunks")
        with col2:
            if st.button("Remove", key=f"remove_{doc['document_id']}"):
                store.remove_document(doc["document_id"])
                st.rerun()

    st.caption(f"Total chunks indexed: {store.total_chunks}")


def render_ask_section() -> None:
    st.subheader("❓ Ask a Question")

    store = get_store()
    documents = store.list_documents()

    if not documents:
        st.info("Upload and process at least one document before asking questions.")
        return

    scope_options = {"All indexed documents": None}
    for doc in documents:
        scope_options[doc["document_name"]] = doc["document_id"]

    scope_label = st.selectbox("Search within", options=list(scope_options.keys()))
    document_id = scope_options[scope_label]

    top_k = st.slider("Number of chunks to retrieve (top-K)", min_value=1, max_value=10, value=config.DEFAULT_TOP_K)

    question = st.text_area("Your question", placeholder="Ask something about the document(s)...")

    if st.button("Ask", type="primary"):
        is_valid, message = validate_question(question)
        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Searching for relevant context..."):
            try:
                chunks = retrieve_chunks(store, question.strip(), top_k=top_k, document_id=document_id)
            except VectorStoreError as exc:
                st.error(f"Retrieval failed: {exc}")
                return
            except Exception:  # noqa: BLE001
                st.error("An unexpected error occurred during retrieval. Please try again.")
                return

        if not chunks:
            st.warning(
                "No relevant content was found for this question in the selected "
                "document(s). Try rephrasing, or check that the right document is selected."
            )
            return

        context = build_context(chunks)

        with st.spinner("Generating a grounded answer..."):
            success, answer = generate_answer(question.strip(), context)

        if success:
            st.success("Answer:")
            st.markdown(answer)
        else:
            st.error(answer)

        st.subheader("📌 Sources")
        for chunk in chunks:
            page_info = f", Page: {chunk['page_number']}" if chunk.get("page_number") else ""
            with st.expander(f"Document: {chunk['document_name']}{page_info}, Chunk: {chunk['chunk_id']}"):
                st.write("**Relevant context:**")
                st.write(chunk["chunk_text"])
                st.caption(f"Similarity score: {chunk['score']:.4f}")


def main() -> None:
    st.title("📚 EduRAG Document Q&A")
    st.caption("Ask questions about your own documents, with answers grounded only in retrieved content.")

    render_api_key_warning()

    st.subheader("📤 Upload a Document")
    st.markdown("Supported formats: **PDF**, **TXT**.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file is not None and st.button("Upload and Process", type="primary"):
        process_upload(uploaded_file)

    st.divider()
    render_document_status()

    st.divider()
    render_ask_section()


if __name__ == "__main__":
    main()
