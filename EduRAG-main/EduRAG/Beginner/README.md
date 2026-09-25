# 🎓 EduRAG StudyMate — AI-Powered Student Learning Assistant

**Level:** Beginner
**Part of:** EduRAG AI Engineer Project Series (Beginner → Intermediate → Advanced)

## 1. Project Overview

EduRAG StudyMate is a simple, practical AI-powered student utility application
built with Streamlit and a local Ollama model. It demonstrates core AI engineering
skills — LLM API integration, structured prompt engineering, input
validation, and error handling — through four real-world study tools.

This is the **Beginner Level** implementation. Intermediate (Document Q&A
with retrieval) and Advanced (multi-agent, production-grade) versions are
planned separately and are intentionally not implemented here.

## 2. Problem Statement

Students often struggle to make the most of their study material: notes are
long and unstructured, difficult concepts are hard to unpack alone, self-testing
material is time-consuming to create, and it's hard to know how to improve a
written answer without feedback. EduRAG StudyMate addresses each of these
problems with a focused AI tool.

## 3. Features

1. **📝 Note Summarizer** — Converts long study notes into a short summary,
   key concepts, important points, and a quick revision section.
2. **💡 Concept Explainer** — Explains any concept with a simple definition,
   step-by-step explanation, real-world analogy, example, and a common
   misconception.
3. **❓ Quiz Generator** — Generates 5 multiple-choice and 2 short-answer
   questions (Easy/Medium/Hard) with an answer key, based only on supplied
   material.
4. **✍️ Answer Improver** — Takes a question and the student's answer, and
   returns an improved version, what was improved, and missing points.

## 4. Technology Stack

- **Python 3.11+**
- **Streamlit** — UI framework
- **Ollama** (via its local REST API, called with `requests`) — LLM provider
- **python-dotenv** — environment variable management

No LangChain, LangGraph, FastAPI, FAISS, databases, or Docker are used in
this Beginner version by design — these are introduced in later levels.

## 5. Architecture

```
User
  ↓
Streamlit UI          (app.py)
  ↓
Input Validation      (validators.py)
  ↓
Prompt Builder        (prompts.py)
  ↓
LLM Service           (llm.py)
  ↓
Response Processing   (app.py)
  ↓
Formatted Output      (app.py)
```

Each layer has a single responsibility, and the LLM integration is isolated
in `llm.py` so the model provider could be swapped later without touching
the UI, prompts, or validation logic.

## 6. Folder Structure

```
EduRAG-AI-Engineer/
│
├── beginner/
│   ├── app.py             # Streamlit UI and page routing
│   ├── llm.py              # Ollama integration and error handling
│   ├── prompts.py          # Structured prompt-building functions
│   ├── validators.py       # Input validation functions
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Example environment variable file (no secrets)
│   ├── .gitignore
│   ├── README.md           # This file
│   └── assets/
│       └── README.md
│
├── intermediate/
│   └── .gitkeep            # Reserved for the Intermediate project
│
└── advanced/
    └── .gitkeep            # Reserved for the Advanced project
```

## 7. Installation Instructions

1. Clone or download this repository and navigate to the `beginner/` folder:
   ```
   cd EduRAG-AI-Engineer/beginner
   ```

2. Install dependencies (see Virtual Environment Setup below first).

## 8. Virtual Environment Setup

```
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

Then install dependencies:

```
pip install -r requirements.txt
```

## 9. Ollama Configuration

1. Install [Ollama](https://ollama.com/download) for your OS and make sure it's
   running:
   ```
   ollama serve
   ```
2. Pull a model (the default used by this app is `llama3.1`):
   ```
   ollama pull llama3.1
   ```
3. Copy `.env.example` to a new file named `.env`:
   ```
   cp .env.example .env
   ```
4. Open `.env` and adjust the values if needed (defaults work for a local
   install):
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1
   ```

No API key is required since Ollama runs locally. `.env` is still listed in
`.gitignore` in case you later point `OLLAMA_BASE_URL` at a remote/hosted
Ollama instance that requires other secrets.

## 10. How to Run the Application

```
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open
it in your browser.

## 11. Explanation of Each Feature

- **Summarize Notes:** Paste notes → click Generate Summary → receive a
  structured summary grounded only in the notes provided.
- **Explain Concept:** Enter a topic → click Explain Concept → receive a
  structured, beginner-friendly explanation.
- **Generate Quiz:** Paste material, choose a difficulty → click Generate
  Quiz → receive multiple-choice and short-answer questions with an answer
  key based only on that material.
- **Improve Answer:** Enter a question and your answer → click Improve
  Answer → receive an improved answer, what was changed, and what was
  missing.

## 12. Prompt Engineering Approach

All prompts live in `prompts.py`, separate from the UI and API code. Each
prompt:

- Assigns the model a clear role ("You are an AI study assistant...").
- States explicit requirements and formatting instructions (fixed section
  headers so output is predictable and easy to render).
- Instructs the model to stay grounded in the user's provided content and
  not invent information not present in the input.
- Preserves the student's original meaning where relevant (Answer Improver).

## 13. Input Validation

Implemented in `validators.py` and always run **before** any API call:

- Empty input (notes, concept, material, question, answer)
- Minimum length checks (too short to be meaningful)
- Maximum length checks (protects against excessive input)
- Valid quiz difficulty check (must be Easy, Medium, or Hard)

If validation fails, a friendly `st.warning()` message is shown and no API
request is made.

## 14. Error Handling

Implemented in `llm.py`, covering:

- Ollama server unreachable (not running / wrong `OLLAMA_BASE_URL`)
- Requested model not pulled on the Ollama server
- Request timeouts
- Empty model responses
- Any other unexpected exception

All failures are converted into a single friendly message shown via
`st.error()` — no raw stack traces are ever shown to the user.

## 15. Security Considerations

- No API key is required for a local Ollama install; configuration (base URL,
  model name) is read only from the environment via `python-dotenv`.
- `.env` is excluded from version control via `.gitignore`.
- Only `.env.example` (with placeholder-safe defaults) is committed.
- User input is validated before use and is not persisted anywhere.
- All prompts and generated text stay on your machine — nothing is sent to
  a third-party API.

## 16. Example Usage

1. Open the app and select **Summarize Notes** from the sidebar.
2. Paste a paragraph of biology notes.
3. Click **Generate Summary**.
4. Review the structured summary, key concepts, important points, and quick
   revision section.

## 17. Future Improvements

- **Intermediate level:** Add PDF/TXT document upload with chunking,
  embeddings, and FAISS-based retrieval for grounded Document Q&A.
- **Advanced level:** Introduce FastAPI backend, LangGraph orchestration,
  query rewriting, reranking, groundedness checking, Docker packaging, and
  automated tests.
- Add conversation history / export of generated study material.
- Add support for additional LLM providers (Gemini, OpenAI, Groq, etc.) via
  the isolated `llm.py` layer.
