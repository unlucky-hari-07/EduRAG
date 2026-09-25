"""
frontend/app.py

EduRAG Pro Streamlit frontend. This file contains ONLY presentation logic -
it talks to the FastAPI backend over HTTP and never touches the retrieval,
generation, or LangGraph internals directly. This keeps frontend and backend
cleanly separated and lets either be deployed/scaled independently.

Run with:
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="EduRAG Pro", page_icon="🤖", layout="wide")


def _get(path: str, timeout: int = 15):
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def _post(path: str, json_body: dict = None, files=None, timeout: int = 60):
    try:
        response = requests.post(f"{BACKEND_URL}{path}", json=json_body, files=files, timeout=timeout)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return False, detail
        return True, response.json()
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def _delete(path: str, timeout: int = 15):
    try:
        response = requests.delete(f"{BACKEND_URL}{path}", timeout=timeout)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return False, detail
        return True, response.json()
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


def page_dashboard() -> None:
    st.header("Dashboard")

    ok, health = _get("/health")
    if not ok:
        st.error(f"Could not reach the backend at {BACKEND_URL}. Is it running? ({health})")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Backend status", health.get("status", "unknown"))
    col2.metric("Ollama available", "Yes" if health.get("ollama_available") else "No")
    col3.metric("Indexed documents", health.get("indexed_documents", 0))

    if not health.get("ollama_available"):
        st.warning(
            "⚠️ The backend could not reach Ollama. Make sure it's installed "
            "and running (`ollama serve`) on the machine hosting the "
            "backend, and that `OLLAMA_BASE_URL`/`OLLAMA_MODEL_NAME` in the "
            "backend's `.env` file are set correctly."
        )


def page_documents() -> None:
    st.header("Documents")

    ok, data = _get("/documents")
    if not ok:
        st.error(f"Could not reach the backend at {BACKEND_URL}. ({data})")
        return

    documents = data.get("documents", [])

    if not documents:
        st.info("No documents indexed yet. Go to the Upload page to add one.")
        return

    for doc in documents:
        with st.expander(f"📄 {doc['document_name']}"):
            st.write(f"**Document ID:** `{doc['document_id']}`")
            st.write(f"**Chunks indexed:** {doc['chunk_count']}")
            st.write(f"**Ingested at:** {doc['ingested_at']}")
            if st.button("Delete", key=f"delete_{doc['document_id']}"):
                ok, result = _delete(f"/documents/{doc['document_id']}")
                if ok:
                    st.success(result.get("message", "Deleted."))
                    st.rerun()
                else:
                    st.error(f"Failed to delete: {result}")


def page_upload() -> None:
    st.header("Upload a Document")
    st.markdown("Supported formats: **PDF**, **TXT**, **Markdown (.md)**.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    if uploaded_file is not None and st.button("Upload and Index", type="primary"):
        with st.spinner("Uploading, extracting, chunking, and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            ok, result = _post("/documents/upload", files=files)

        if ok:
            st.success(
                f"'{result['document_name']}' indexed successfully "
                f"({result['chunk_count']} chunks)."
            )
        else:
            st.error(f"Upload failed: {result}")


def page_ask_questions() -> None:
    st.header("Ask Questions")

    ok, data = _get("/documents")
    documents = data.get("documents", []) if ok else []

    scope_options = {"All documents": None}
    for doc in documents:
        scope_options[doc["document_name"]] = doc["document_id"]

    scope_label = st.selectbox("Document scope", options=list(scope_options.keys()))
    document_id = scope_options[scope_label]

    question = st.text_area("Your question", placeholder="Ask something about your documents...")

    if st.button("Ask", type="primary"):
        question_clean = question.strip()
        if not question_clean:
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking through your documents..."):
            ok, result = _post("/chat", json_body={"question": question_clean, "document_id": document_id})

        if not ok:
            st.error(f"Request failed: {result}")
            return

        if result.get("insufficient_context"):
            st.warning(result["answer"])
        else:
            st.success("Answer:")
            st.markdown(result["answer"])

        with st.expander("Retrieval & groundedness details"):
            st.write(f"**Retrieved candidates:** {result['retrieved_chunks']}")
            st.write(f"**Chunks after reranking:** {result['reranked_chunks']}")
            st.write(f"**Groundedness check passed:** {'✅ Yes' if result['grounded'] else '⚠️ No'}")
            st.write(f"**Refinement iterations used:** {result['iterations']}")
            if result.get("unsupported_claims"):
                st.write("**Unsupported claims flagged:**")
                for claim in result["unsupported_claims"]:
                    st.write(f"- {claim}")

        if result.get("sources"):
            st.subheader("Sources")
            for source in result["sources"]:
                page_info = f", page {source['page_number']}" if source.get("page_number") else ""
                with st.expander(f"📄 {source['document_name']}{page_info} — chunk {source['chunk_id']}"):
                    st.write(source["text"])


def main() -> None:
    st.title("🤖 EduRAG Pro")
    st.caption("Production-style agentic document intelligence system")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", options=["Dashboard", "Documents", "Upload", "Ask Questions"])

    pages = {
        "Dashboard": page_dashboard,
        "Documents": page_documents,
        "Upload": page_upload,
        "Ask Questions": page_ask_questions,
    }
    pages[page]()


if __name__ == "__main__":
    main()
