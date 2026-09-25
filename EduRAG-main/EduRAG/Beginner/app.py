"""
app.py

EduRAG StudyMate - AI-Powered Student Learning Assistant (Beginner Level)

Application flow:

    User
      |
    Streamlit UI          (this file)
      |
    Input Validation       (validators.py)
      |
    Prompt Builder         (prompts.py)
      |
    LLM Service            (llm.py)
      |
    Response Processing    (this file)
      |
    Formatted Output       (this file)

Run with:
    streamlit run app.py
"""

import streamlit as st

from llm import generate_response, is_api_key_configured
from prompts import (
    build_answer_improvement_prompt,
    build_concept_explanation_prompt,
    build_quiz_prompt,
    build_summary_prompt,
)
from validators import (
    validate_answer_improvement,
    validate_concept,
    validate_notes,
    validate_quiz_material,
)

st.set_page_config(page_title="EduRAG StudyMate", page_icon="🎓", layout="centered")


def render_result(success: bool, content: str) -> None:
    """Display the LLM result (or error) in a consistent, readable way."""
    if success:
        st.success("Done! Here's your result:")
        st.markdown(content)
    else:
        st.error(content)


def render_api_key_warning() -> None:
    """Show a persistent warning banner if no API key is configured."""
    if not is_api_key_configured():
        st.warning(
            "⚠️ Could not reach Ollama. Make sure it's installed and running "
            "(`ollama serve`), and that `OLLAMA_BASE_URL`/`OLLAMA_MODEL` in "
            "your `.env` file are set correctly. See `.env.example` for the "
            "expected format."
        )


def page_home() -> None:
    st.header("Welcome 👋")
    st.markdown(
        """
EduRAG StudyMate is an AI-powered learning assistant built to help students
study more effectively. Use the navigation menu on the left to choose a tool:

- **📝 Summarize Notes** — turn long notes into a clear, organized summary.
- **💡 Explain Concept** — get a simple, structured explanation of any topic.
- **❓ Generate Quiz** — turn your study material into a practice quiz.
- **✍️ Improve Answer** — get feedback and an improved version of your answer.

All tools are powered by a local Ollama model and only use the information
you provide — the AI is instructed not to invent facts beyond your input.
        """
    )


def page_summarize_notes() -> None:
    st.header("📝 Summarize Notes")
    st.markdown(
        "Paste your study notes below. The AI will produce a short summary, "
        "key concepts, important points, and a quick revision section — "
        "based only on what you provide."
    )

    notes = st.text_area("Your study notes", height=250, placeholder="Paste your notes here...")

    if st.button("Generate Summary", type="primary"):
        is_valid, message = validate_notes(notes)

        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Summarizing your notes..."):
            prompt = build_summary_prompt(notes.strip())
            success, content = generate_response(prompt)

        render_result(success, content)


def page_explain_concept() -> None:
    st.header("💡 Explain Concept")
    st.markdown(
        "Enter a concept or topic you'd like explained. You'll get a simple "
        "definition, a step-by-step explanation, a real-world analogy, an "
        "example, and a common misconception to watch out for."
    )

    concept = st.text_input("Concept to explain", placeholder="e.g. Newton's Third Law")

    if st.button("Explain Concept", type="primary"):
        is_valid, message = validate_concept(concept)

        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Preparing your explanation..."):
            prompt = build_concept_explanation_prompt(concept.strip())
            success, content = generate_response(prompt)

        render_result(success, content)


def page_generate_quiz() -> None:
    st.header("❓ Generate Quiz")
    st.markdown(
        "Paste your study material and choose a difficulty. The AI will "
        "generate 5 multiple-choice questions and 2 short-answer questions "
        "with an answer key — based only on the material you provide."
    )

    material = st.text_area(
        "Study material", height=250, placeholder="Paste the material to be quizzed on..."
    )
    difficulty = st.selectbox("Difficulty", options=["Easy", "Medium", "Hard"])

    if st.button("Generate Quiz", type="primary"):
        is_valid, message = validate_quiz_material(material, difficulty)

        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Generating your quiz..."):
            prompt = build_quiz_prompt(material.strip(), difficulty)
            success, content = generate_response(prompt)

        render_result(success, content)


def page_improve_answer() -> None:
    st.header("✍️ Improve Answer")
    st.markdown(
        "Enter the question you were asked and the answer you wrote. The AI "
        "will suggest an improved version, explain what was improved, and "
        "point out any missing points — without changing your original intent."
    )

    question = st.text_input("Question", placeholder="e.g. What causes seasons on Earth?")
    answer = st.text_area("Your answer", height=180, placeholder="Paste your answer here...")

    if st.button("Improve Answer", type="primary"):
        is_valid, message = validate_answer_improvement(question, answer)

        if not is_valid:
            st.warning(message)
            return

        with st.spinner("Improving your answer..."):
            prompt = build_answer_improvement_prompt(question.strip(), answer.strip())
            success, content = generate_response(prompt)

        render_result(success, content)


def main() -> None:
    st.title("🎓 EduRAG StudyMate")
    st.caption("AI-Powered Student Learning Assistant")

    render_api_key_warning()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=[
            "Home",
            "Summarize Notes",
            "Explain Concept",
            "Generate Quiz",
            "Improve Answer",
        ],
    )

    pages = {
        "Home": page_home,
        "Summarize Notes": page_summarize_notes,
        "Explain Concept": page_explain_concept,
        "Generate Quiz": page_generate_quiz,
        "Improve Answer": page_improve_answer,
    }

    pages[page]()


if __name__ == "__main__":
    main()
