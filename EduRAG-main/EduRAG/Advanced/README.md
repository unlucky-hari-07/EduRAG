# 🤖 EduRAG Pro — Production-Style Agentic Document Intelligence System

**Level:** Advanced
**Part of:** EduRAG AI Engineer Project Series (Beginner → Intermediate → Advanced)

## 1. Project Overview

EduRAG Pro lets a user upload documents (PDF, TXT, Markdown), ask questions
about them, and receive answers that are grounded in the retrieved content,
cited to their exact source chunks, and verified for support before being
returned. It is built as a modular FastAPI backend orchestrated by a
LangGraph agentic workflow, with a Streamlit frontend as a thin HTTP client.

## 2. Problem

Plain LLM chat over documents has two common failure modes: (1) naive
retrieval returns loosely related chunks, producing answers that miss the
point, and (2) the model answers confidently even when the retrieved
content doesn't actually support the claim ("hallucination"). EduRAG Pro
addresses both with a multi-stage retrieval pipeline (retrieve → rerank →
context-quality check) and an explicit post-generation groundedness check
with a bounded refine-and-retry loop.

## 3. Why RAG

Retrieval-Augmented Generation grounds the LLM's answer in content it did
not memorize during training, which matters here because: the documents are
user-supplied and unknown ahead of time, answers need traceable citations
back to the source, and hallucination risk should be reduced by constraining
generation to retrieved text rather than open-ended model knowledge.

## 4. Architecture

```
Streamlit Frontend
        |
   FastAPI Backend
        |
  Pydantic Validation
        |
  LangGraph Workflow
        |
   Query Validation
        |
   Query Rewriting
        |
   Document Scope
        |
     Retriever (FAISS)
        |
     Reranker (cross-encoder)
        |
  Context Quality Check
        |
  Grounded Answer Generation
        |
   Groundedness Check
        |
 Final Structured Response
```

If context is insufficient or the answer fails the groundedness check:

```
Context Check / Groundedness Check
        |
   Query Refinement
        |
   Retrieval Again  (bounded by MAX_GRAPH_ITERATIONS)
        |
    Generation
```

## 5. Component Responsibilities

| Layer | Responsibility |
|---|---|
| `frontend/app.py` | Streamlit UI; talks to the backend only over HTTP |
| `backend/main.py` | FastAPI app, router wiring, CORS, error handling |
| `backend/api/` | HTTP route handlers (documents, chat, health) |
| `backend/models/` | Pydantic request/response schemas |
| `backend/services/` | Embeddings, ingestion orchestration, generation, grounding |
| `ingestion/` | Pure text extraction, cleaning, and chunking logic |
| `retrieval/` | FAISS index management, search, cross-encoder reranking |
| `graph/` | LangGraph state, nodes, and workflow assembly |
| `prompts/` | Externalized prompt templates |
| `evaluation/` | Small, honest retrieval/grounding evaluation scripts |
| `tests/` | API and validation test suites |

## 6. API Architecture

- `POST /documents/upload` — validate, extract, chunk, embed, index, and
  register a document.
- `GET /documents` — list indexed documents with chunk counts.
- `DELETE /documents/{document_id}` — remove a document's chunks from the
  FAISS index and its registry entry.
- `POST /chat` — run a question through the LangGraph workflow and return a
  structured, cited answer.
- `GET /health` — service status, Ollama availability, and index size.

All requests and responses are validated via Pydantic models
(`backend/models/requests.py`, `backend/models/responses.py`). Validation
failures return `422` with a readable `detail` message; unexpected server
errors return `500` with a generic message — no internal stack traces are
ever returned to the client.

## 7. LangGraph Workflow

Typed state (`graph/state.py`) is threaded through nine nodes
(`graph/nodes.py`), assembled with conditional edges in
`graph/workflow.py`:

```
validate_query → rewrite_query → retrieve → rerank → check_context
  check_context:
    sufficient        → generate_answer
    insufficient
      + retries left  → refine_query → retrieve   (loop)
      no retries left → finalize
  generate_answer → check_groundedness
  check_groundedness:
    grounded          → finalize
    not grounded
      + retries left  → refine_query → retrieve   (loop)
      no retries left → finalize
  finalize → END
```

`refine_query` increments an `iteration` counter compared against
`max_iterations` (`MAX_GRAPH_ITERATIONS`, default 2) in both conditional
routers — this makes an infinite loop structurally impossible, not just
unlikely.

## 8. Query Rewriting

`backend/services/generation.rewrite_query()` sends the raw question through
`prompts/query_rewrite.txt`, which instructs the model to make vague or
short questions more retrieval-friendly while explicitly preserving the
user's original intent and never introducing new facts. If rewriting fails
(e.g. API error), the pipeline fails open and uses the original question
rather than blocking.

## 9. Retrieval

`retrieval/index.py` wraps a FAISS `IndexFlatIP` (cosine similarity via
L2-normalized vectors) with a JSON metadata sidecar tracking
`document_id`, `document_name`, `page_number`, `chunk_id`, and `chunk_text`
for every vector — so nothing about a source is ever inferred or invented
later. `retrieval/search.py` embeds the (rewritten) query and retrieves
`RETRIEVAL_CANDIDATE_K` (default 15) candidates, optionally scoped to one
document by over-fetching and filtering.

## 10. Reranking

`retrieval/rerank.py` re-scores the FAISS candidates with a
Sentence-Transformers cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`
by default), keeping the top `RERANK_TOP_K` (default 5). A cross-encoder
reads the query and each candidate chunk *together*, which tends to judge
relevance more precisely than comparing independent embedding vectors — at
the cost of being too slow to run over an entire corpus, which is why it
only re-scores the small candidate set FAISS already narrowed down. **No
formal benchmark has been run comparing reranked vs. non-reranked precision
in this project** — this is an architectural justification, not a measured
result.

## 11. Context Quality Checking

`graph/nodes.check_context` looks at the best cross-encoder score among the
reranked chunks. If it falls below `MIN_CONTEXT_SCORE` (default 0.15), the
context is treated as insufficient and the workflow refines the query and
retries (up to the iteration limit) rather than generating from weak
context.

## 12. Grounded Generation

`prompts/answer.txt` instructs the model to answer using **only** the
supplied context, explicitly state when the context is insufficient rather
than guess, and never invent facts, citations, page numbers, or sources —
source metadata is attached by the application, not generated by the model.

## 13. Groundedness Verification

`backend/services/grounding.check_groundedness()` sends the question,
context, and generated answer through `prompts/groundedness.txt` and parses
a structured `{"grounded": bool, "unsupported_claims": [...], "reason": "..."}`
verdict (robust to the model wrapping it in markdown fences). If the
response can't be parsed, or the Ollama call itself fails, the check **fails
safe** — it returns `grounded=false` rather than assuming the answer is
fine.

## 14. Source Citation Mechanism

Every `SourceInfo` returned in a `/chat` response (`document_id`,
`document_name`, `page_number`, `chunk_id`, `text`) is built directly from
the metadata stored at ingestion time (`graph/nodes.rerank`) — never from
LLM output. The model never sees or produces page numbers or document
names, so a citation cannot be fabricated.

## 15. Hallucination Mitigation

This system reduces hallucination risk through three layered mechanisms:
constraining generation to retrieved context, a context-sufficiency check
before generation, and a post-hoc groundedness check with a bounded
refine-and-retry loop. **This is a mitigation strategy, not a guarantee.**
The groundedness check itself is performed by an LLM and can be wrong; no
claim of mathematically guaranteed hallucination elimination is made
anywhere in this system.

## 16. Pydantic Validation

Used throughout `backend/models/`: `ChatRequest` (question length bounds,
non-blank `document_id`), `ChatResponse`, `SourceInfo`, `DocumentInfo`,
`UploadResponse`, `DeleteResponse`, `HealthResponse`, and `ErrorResponse`.
No unvalidated dictionaries cross the API boundary in either direction.

## 17. FastAPI Endpoints

See section 6 above for the full list. A custom
`RequestValidationError` handler in `backend/main.py` converts Pydantic
validation failures into a clean `{"detail": "..."}` 422 response.

## 18. Streamlit Frontend

`frontend/app.py` has four pages — Dashboard, Documents, Upload, Ask
Questions — and communicates with the backend exclusively via `requests`
HTTP calls to `BACKEND_URL`. It contains no retrieval, generation, or graph
logic of its own, so backend and frontend can be developed, tested, and
scaled independently.

## 19. Docker Setup

A single `Dockerfile` is shared by both services; `docker-compose.yml` runs
three containers from it/alongside it: `ollama` (the local LLM server),
`backend`, and `frontend`.

```bash
# 1. Copy .env.example to .env (defaults work out of the box for Ollama)
cp .env.example .env

# 2. Build and start all services
docker compose up --build

# 3. Pull a model into the ollama container (one-time, or after a fresh volume)
docker compose exec ollama ollama pull llama3.1

# Backend:  http://localhost:8000
# Frontend: http://localhost:8501
# Ollama:   http://localhost:11434
```

No API key or secret is baked into the image — `docker-compose.yml` loads
`backend`'s environment from `.env` (excluded from git) via `env_file`,
points it at the `ollama` service via `OLLAMA_BASE_URL=http://ollama:11434`,
and points the `frontend` service at the backend container via
`BACKEND_URL=http://backend:8000`.

## 20. Testing

```bash
pip install -r requirements.txt
pytest tests/
```

- `tests/test_validation.py` — Pydantic `ChatRequest` validation and
  document upload validation (file type, size, emptiness). Runs offline,
  no Ollama server needed.
- `tests/test_api.py` — FastAPI endpoint tests via `TestClient`: health
  check, invalid chat requests (empty/short/missing question), invalid
  document uploads (unsupported type, empty file), document listing shape,
  and a 404 on deleting a nonexistent document. One end-to-end
  upload-then-chat test is included but automatically skipped unless a
  running Ollama server (with a pulled model) is detected, since it calls
  the live Ollama API.

**Honesty note on execution:** this project was developed in a sandboxed
environment with no internet access, so `pytest` itself and the ML
dependencies (FastAPI, Pydantic, LangGraph, FAISS, Sentence Transformers)
could not be installed or executed there. Every file was syntax-checked
(`python -m py_compile`), and all pure-Python logic that doesn't require
those packages — chunking, cleaning, loading, upload validation, and the
groundedness JSON-parsing logic — was manually unit-tested with passing
results. Run `pytest tests/` locally to confirm the full suite, including
the parts that need the real dependencies.

## 21. Evaluation Methodology

`evaluation/dataset.json` holds a small example set of
question/expected-document/expected-concepts triples. `evaluation/test_retrieval.py`
checks whether the expected document appears among retrieved sources;
`evaluation/test_grounding.py` checks whether the `grounded` and
`insufficient_context` flags match expectations for each case. Both call
the live `/chat` endpoint and require the backend running with the
referenced documents already uploaded:

```bash
python evaluation/test_retrieval.py
python evaluation/test_grounding.py
```

**No benchmark has actually been executed against this dataset as part of
building this project** — the example `dataset.json` references a
placeholder document name that doesn't exist yet. Replace it with your own
documents and questions, upload them, then run the scripts to get real
numbers for your data. No numeric accuracy or hallucination-rate figure
should be assumed from this repository beyond what you personally run and
observe.

## 22. Limitations

- The FAISS index is a single flat index; document-scoped search filters
  post-hoc rather than using a per-document index, which is fine at small
  scale but wouldn't scale to a very large document collection.
- The groundedness check is itself LLM-based and can be wrong in either
  direction (false positive or false negative).
- No streaming generation (see the note in `backend/services/generation.py`)
  — a deliberate scope decision, not a technical limitation.
- Deleting a document rebuilds the FAISS index in-process, which is fine
  for a project of this size but not for a large, high-throughput index.
- No authentication/authorization on the API — appropriate for a local
  demo/project, not for a multi-tenant production deployment.

## 23. Future Improvements

- Real benchmark run with a larger evaluation set and human-labeled
  relevance judgments.
- Streaming answer generation with incremental UI updates.
- Per-document or sharded FAISS indexes for larger corpora.
- Authentication and per-user document scoping.
- Swap the flat rebuild-on-delete index strategy for a vector store with
  native delete support.

## Folder Structure

```
advanced/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── routes_documents.py
│   │   ├── routes_chat.py
│   │   └── routes_health.py
│   ├── models/
│   │   ├── requests.py
│   │   └── responses.py
│   └── services/
│       ├── ingestion.py
│       ├── embeddings.py
│       ├── generation.py
│       └── grounding.py
├── frontend/
│   └── app.py
├── graph/
│   ├── state.py
│   ├── nodes.py
│   └── workflow.py
├── ingestion/
│   ├── loaders.py
│   ├── cleaner.py
│   └── chunker.py
├── retrieval/
│   ├── index.py
│   ├── search.py
│   └── rerank.py
├── prompts/
│   ├── query_rewrite.txt
│   ├── answer.txt
│   └── groundedness.txt
├── evaluation/
│   ├── dataset.json
│   ├── test_retrieval.py
│   └── test_grounding.py
├── tests/
│   ├── test_api.py
│   └── test_validation.py
├── data/                  # runtime: uploads, FAISS index, documents.json (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Running Locally Without Docker

```bash
cd advanced
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # defaults work if Ollama is running locally

# Make sure Ollama is installed, running, and has the model pulled:
#   ollama serve
#   ollama pull llama3.1

# Terminal 1
uvicorn backend.main:app --reload --port 8000

# Terminal 2
streamlit run frontend/app.py
```
