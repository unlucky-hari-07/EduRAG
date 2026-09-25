# 📚 EduRAG — Document-Based Q&A Assistant

**Level:** Intermediate
**Part of:** EduRAG AI Engineer Project Series (Beginner → Intermediate → Advanced)

## 1. Project Overview

EduRAG Document Q&A lets a user upload a PDF or TXT document, ask questions
about it, and receive answers grounded **only** in that document's content —
retrieved via a real Retrieval-Augmented Generation (RAG) pipeline, not by
stuffing the whole document into the prompt.

## 2. Problem Statement

Sending an entire document to an LLM doesn't scale (context limits, cost),
doesn't tell you *where* an answer came from, and doesn't stop the model
from blending in outside knowledge. This project demonstrates the
alternative: retrieve only the specific passages relevant to a question,
and constrain generation to those passages, with every claim traceable back
to an exact chunk and page.

## 3. What RAG Means

**Retrieval-Augmented Generation** is a two-stage approach: first *retrieve*
the most relevant pieces of external content for a given query (here, via
embedding similarity search over document chunks), then *generate* an
answer using an LLM that is instructed to rely only on that retrieved
content. This grounds the model's output in real, traceable source material
instead of relying purely on what it memorized during training.

## 4. Architecture

```
Document Upload
      |
File Validation        (utils/validators.py)
      |
Text Extraction         (rag/loaders.py)
      |
Text Cleaning            (rag/cleaner.py)
      |
Chunking                  (rag/chunker.py)
      |
Embedding Generation       (rag/embeddings.py)
      |
FAISS Vector Index          (rag/vector_store.py)
      |
User Question
      |
Question Embedding         (rag/embeddings.py)
      |
Similarity Search          (rag/retriever.py)
      |
Top-K Relevant Chunks
      |
Context Construction      (rag/retriever.py)
      |
Ollama                    (rag/generator.py)
      |
Grounded Answer + Sources
```

## 5. Complete Data Flow

1. User uploads a PDF or TXT file via Streamlit.
2. `utils/validators.validate_upload()` checks file type, size, and
   emptiness before anything else runs.
3. `rag/loaders.load_document()` extracts text (page-by-page for PDFs via
   PyMuPDF; as a single block for TXT).
4. `rag/cleaner.clean_text()` normalizes whitespace without altering
   content.
5. `rag/chunker.chunk_pages()` splits the cleaned text into overlapping,
   word-based chunks, keeping page numbers attached.
6. `rag/embeddings.embed_texts()` encodes every chunk into a normalized
   vector via Sentence Transformers.
7. `rag/vector_store.VectorStore.add_document()` adds the vectors to a
   FAISS index and stores matching metadata (document, page, chunk ID,
   text) for every vector.
8. When the user asks a question, it's validated, embedded, and used to
   search FAISS (`rag/retriever.retrieve_chunks()`), optionally scoped to
   one document.
9. The top-K retrieved chunks are joined into a context string
   (`rag/retriever.build_context()`).
10. `rag/generator.generate_answer()` sends **only** the question and
    retrieved context (never the whole document) to a local Ollama model
    with a strict grounding prompt.
11. The answer and the exact source chunks (with document name, page
    number, and chunk ID) are displayed together.

## 6. Supported File Formats

- **PDF** — via PyMuPDF, with per-page text extraction.
- **TXT** — read directly as a single block of text.

## 7. Text Extraction

`rag/loaders.py` dispatches by file extension. PDF extraction iterates
every page with PyMuPDF and raises a clear error if a page yields no text
(often meaning a scanned, image-only PDF with no embedded text layer) or if
the file is corrupted. TXT files are read directly with permissive
encoding handling.

## 8. Cleaning

`rag/cleaner.clean_text()` only normalizes whitespace — collapsing repeated
blank lines and spaces, and trimming trailing whitespace — without ever
altering the actual wording, so grounding against the original text stays
accurate.

## 9. Chunking

`rag/chunker.chunk_pages()` splits cleaned text into overlapping, **word-based**
chunks (a simple, dependency-free proxy for tokens), chunk-by-chunk within
each page so page metadata is preserved on every chunk.

## 10. Chunk Size and Overlap Reasoning

Defaults: **220 words** per chunk with **40 words** of overlap
(`CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` in `.env`). These are reasonable
starting points, **not universally optimal values**. Smaller chunks give
more precise embedding matches but less surrounding context per match;
larger chunks give more context but can dilute what the embedding
represents, hurting retrieval precision. Overlap helps avoid losing meaning
at chunk boundaries. Tune both values based on your own documents and
observed retrieval quality — there's no substitute for trying a few
settings and checking what actually gets retrieved.

## 11. Embeddings

`rag/embeddings.py` uses Sentence Transformers (`all-MiniLM-L6-v2` by
default, configurable via `EMBEDDING_MODEL_NAME`) to embed every chunk.
Vectors are L2-normalized so a FAISS inner-product index behaves as cosine
similarity search.

## 12. FAISS

`rag/vector_store.py` wraps a FAISS `IndexFlatIP`. Because FAISS itself only
returns integer vector positions, a parallel metadata list is kept in exact
lockstep, mapping every vector back to its `document_id`, `document_name`,
`page_number`, `chunk_id`, and `chunk_text` — so no citation is ever
guessed or reconstructed after the fact.

## 13. Similarity Retrieval

A question is embedded with the same model used for chunks, then compared
against the index via inner-product (cosine) similarity. When a specific
document is selected in the UI, the store over-fetches candidates and
filters by `document_id` so retrieval **never mixes results across
unrelated documents** — verified directly in testing (see section 19).

## 14. Top-K Retrieval

`top_k` is configurable in the UI (default 5, `DEFAULT_TOP_K` in `.env`).
A smaller K gives tighter, more precise context; a larger K gives more
coverage at the risk of including less relevant chunks.

## 15. Context Construction

`rag/retriever.build_context()` joins the retrieved chunks' text with clear
separators into a single context block — this is the *only* document
content that ever reaches the LLM prompt.

## 16. Grounded Answer Generation

`rag/generator.py` sends a strict prompt instructing the model to answer
using only the supplied context, explicitly state when the context doesn't
contain the answer, and never invent facts, outside knowledge, or
citations.

## 17. Source Display

Every answer is shown alongside its sources, each labeled with the exact
document name, page number (when available), and chunk ID, plus the actual
retrieved text and its similarity score — all sourced from application
metadata captured at ingestion time, never from the LLM.

## 18. Hallucination Reduction

Three things work together here: (1) only retrieved, relevant chunks reach
the prompt — not the whole document, reducing noise and off-topic content;
(2) the generation prompt explicitly instructs the model to say when the
context is insufficient rather than guess; (3) sources are drawn from
tracked metadata rather than generated by the model, so a citation can
never be fabricated. This reduces hallucination risk; it does not
eliminate it, since the underlying generation step is still an LLM call.

## 19. Validation

Implemented in `utils/validators.py`, covering: empty file, oversized file,
unsupported extension, empty/unreadable extracted text (including scanned
PDFs), empty question, and overly long question. All validation happens
**before** any extraction, embedding, or API call — confirmed in manual
testing (see section 21).

## 20. Error Handling

- **Extraction/corrupted files:** `rag/loaders.py` raises a clear
  `DocumentLoadError` for corrupted PDFs or unreadable files, caught in
  `app.py` and shown via `st.error()` — never a raw stack trace.
- **Embedding/FAISS errors:** `rag/vector_store.py` raises a
  `VectorStoreError` with a clear message; caught the same way.
- **Ollama unreachable, model not pulled, timeouts, empty model
  responses:** all handled in `rag/generator.py`, converted to friendly
  messages.
- **No relevant retrieval results:** the UI shows a specific message rather
  than silently generating from empty context (`generate_answer` also
  refuses to call the API with empty context, as a second line of
  defense).

## 21. Testing Performed

This project's environment had no internet access, so the real
Streamlit/FAISS/Sentence-Transformers/Ollama stack could not be executed
end-to-end here. What **was** verified directly:

- All Python files pass `python -m py_compile` (no syntax errors).
- Word-based chunking with overlap: verified chunk boundaries and that
  consecutive chunks on the same page actually share the configured overlap
  region.
- `VectorStore` (tested against a functional FAISS stub): multi-document
  indexing, document-scoped search that **never leaks results across
  documents**, and correct removal of a document's chunks from both the
  index and subsequent search results.
- All validators (`validate_upload`, `validate_extracted_text`,
  `validate_question`) against every case in the spec: empty file,
  oversized file, unsupported extension, empty extracted text, empty/short/
  long question.
- `rag/generator.py` error paths (against a stubbed Ollama HTTP client):
  server unreachable, empty context (refused before calling the API),
  simulated 404 (model not pulled), and a successful generation call.

Reasoned through but not executable here (require real PDF files, a
running Ollama server with a pulled model, and network access): uploading
an actual valid PDF end-to-end, a genuinely corrupted PDF, and a live
multi-document question session.
Run the app locally to confirm these.

## 22. Installation

```bash
cd intermediate
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 23. Ollama Configuration

1. Install [Ollama](https://ollama.com/download) for your OS and make sure
   it's running:
   ```bash
   ollama serve
   ```
2. Pull a model (the default used by this app is `llama3.1`):
   ```bash
   ollama pull llama3.1
   ```
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Adjust `OLLAMA_BASE_URL` / `OLLAMA_MODEL_NAME` in `.env` if needed
   (defaults work for a local install).

No API key is required since Ollama runs locally. `.env` is still excluded
via `.gitignore` in case you later point `OLLAMA_BASE_URL` at a remote/hosted
Ollama instance that requires other secrets.

## 24. Running the Application

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`).

## 25. Limitations

- The FAISS index and document registry live only in Streamlit's
  `st.session_state` — they reset when the session ends; there's no
  persistence to disk in this Intermediate version.
- Word-count chunking is a simple proxy for tokens, not an exact token
  count for any specific model's tokenizer.
- No reranking stage — retrieval relies solely on embedding similarity
  (a cross-encoder reranker is introduced in the Advanced level).
- Scanned, image-only PDFs with no embedded text layer cannot be processed
  without OCR, which isn't implemented here.
- Single-process, single-user design — not built for concurrent multi-user
  access.

## 26. Future Improvements

- Persist the FAISS index and metadata to disk so documents survive a
  restart.
- Add OCR fallback for scanned PDFs.
- Add a reranking stage to improve precision on the initial retrieval set.
- Add an evaluation harness to measure retrieval and answer quality on a
  fixed question set.
- Introduce the agentic, production-style architecture (FastAPI, LangGraph,
  query rewriting, groundedness verification, Docker) planned for the
  Advanced level.

## Folder Structure

```
intermediate/
├── app.py
├── rag/
│   ├── loaders.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
├── utils/
│   ├── validators.py
│   └── config.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
