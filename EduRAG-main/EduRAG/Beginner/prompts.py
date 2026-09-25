"""
prompts.py

Structured prompt-building functions for EduRAG StudyMate.

Each function takes the user's raw input and returns a single, fully-formed
prompt string ready to send to the LLM. Keeping prompts here (rather than
inline in app.py) makes them easy to review, version, and improve without
touching the UI or the API integration code.
"""


def build_summary_prompt(notes: str) -> str:
    """Build the prompt for the Note Summarizer feature."""
    return f"""You are an AI study assistant helping a student review their notes.

Summarize the student's notes below.

Requirements:
- Identify the main concepts.
- Preserve important facts and definitions exactly as given.
- Remove unnecessary repetition and filler text.
- Use simple, clear language appropriate for a student.
- Organize your response using these exact section headers:

## Short Summary
## Key Concepts
## Important Points
## Quick Revision

- Do not introduce any information that is not contained in the notes below.
- If the notes are unclear or incomplete, summarize only what is present.

Student notes:
\"\"\"
{notes}
\"\"\"
"""


def build_concept_explanation_prompt(concept: str) -> str:
    """Build the prompt for the Concept Explainer feature."""
    return f"""You are an AI study assistant helping a student understand a concept.

Explain the concept below in a way that is clear and appropriate for a student
who is encountering it for the first time.

Organize your response using these exact section headers:

## Simple Definition
## Step-by-Step Explanation
## Real-World Analogy
## Example
## Common Misconception

Requirements:
- Keep the language simple and beginner-friendly.
- The step-by-step explanation should build understanding gradually.
- The analogy should relate to everyday life.
- The example should be concrete, not abstract.
- The common misconception section should clarify a mistake students often make.

Concept:
\"\"\"
{concept}
\"\"\"
"""


def build_quiz_prompt(material: str, difficulty: str) -> str:
    """Build the prompt for the Quiz Generator feature."""
    return f"""You are an AI study assistant creating a practice quiz for a student.

Base the quiz ONLY on the study material provided below. Do not introduce
facts, names, or concepts that are not present in the material.

Difficulty level: {difficulty}

Generate:
- 5 multiple-choice questions, each with 4 answer options labeled A-D.
- 2 short-answer questions.
- The correct answer clearly marked for every question.
- A brief (1-2 sentence) explanation for each correct answer, referencing the
  material.

Organize your response using these exact section headers:

## Multiple Choice Questions
## Short Answer Questions
## Answer Key

Format each multiple-choice question like this:
1. Question text
   A. Option
   B. Option
   C. Option
   D. Option

List short-answer questions as a simple numbered list.
In the Answer Key section, list the correct answer and a brief explanation
for every question (both multiple-choice and short-answer), numbered to
match the questions above.

Study material:
\"\"\"
{material}
\"\"\"
"""


def build_answer_improvement_prompt(question: str, answer: str) -> str:
    """Build the prompt for the Answer Improver feature."""
    return f"""You are an AI study assistant helping a student improve their answer
to a question.

Question:
\"\"\"
{question}
\"\"\"

Student's answer:
\"\"\"
{answer}
\"\"\"

Requirements:
- Preserve the original meaning and intent of the student's answer.
- Do not change the student's position or introduce a completely different answer.
- Improve clarity, completeness, structure, and accuracy where needed.

Organize your response using these exact section headers:

## Improved Answer
## What Was Improved
## Missing Points

- "What Was Improved" should briefly list the specific changes made (e.g.
  clarity, structure, missing detail, terminology).
- "Missing Points" should list any important points the original answer
  did not cover, based only on the question asked.
"""
