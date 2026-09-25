"""
validators.py

Input validation logic for EduRAG StudyMate.

All validation happens BEFORE any call to the LLM service. Each function
returns a tuple of (is_valid: bool, message: str). If is_valid is False,
`message` explains what the user should fix. If is_valid is True, `message`
is an empty string.

Keeping validation separate from the UI and the LLM logic makes it easy to
test and reuse across features.
"""

from typing import Tuple

# Reasonable bounds for text inputs. These protect the user from empty or
# accidentally huge submissions, and protect the app from wasting API calls
# on inputs that could never produce a useful response.
MIN_NOTES_LENGTH = 20
MAX_NOTES_LENGTH = 8000

MIN_CONCEPT_LENGTH = 2
MAX_CONCEPT_LENGTH = 200

MIN_QUIZ_MATERIAL_LENGTH = 20
MAX_QUIZ_MATERIAL_LENGTH = 8000

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 1000

MIN_ANSWER_LENGTH = 1
MAX_ANSWER_LENGTH = 3000

VALID_QUIZ_DIFFICULTIES = {"Easy", "Medium", "Hard"}


def validate_notes(notes: str) -> Tuple[bool, str]:
    """Validate study notes submitted to the Note Summarizer."""
    notes = notes.strip()

    if not notes:
        return False, "Please enter your study notes before summarizing."

    if len(notes) < MIN_NOTES_LENGTH:
        return False, (
            f"Your notes are too short (minimum {MIN_NOTES_LENGTH} characters). "
            "Please provide more detail so the summary is meaningful."
        )

    if len(notes) > MAX_NOTES_LENGTH:
        return False, (
            f"Your notes are too long (maximum {MAX_NOTES_LENGTH} characters). "
            "Please shorten them or split them into smaller sections."
        )

    return True, ""


def validate_concept(concept: str) -> Tuple[bool, str]:
    """Validate a concept name submitted to the Concept Explainer."""
    concept = concept.strip()

    if not concept:
        return False, "Please enter a concept you'd like explained."

    if len(concept) < MIN_CONCEPT_LENGTH:
        return False, "Please enter a more complete concept name."

    if len(concept) > MAX_CONCEPT_LENGTH:
        return False, (
            f"That concept description is too long (maximum {MAX_CONCEPT_LENGTH} "
            "characters). Try summarizing it into a shorter topic or phrase."
        )

    return True, ""


def validate_quiz_material(material: str, difficulty: str) -> Tuple[bool, str]:
    """Validate study material and difficulty for the Quiz Generator."""
    material = material.strip()

    if not material:
        return False, "Please enter the study material to generate a quiz from."

    if len(material) < MIN_QUIZ_MATERIAL_LENGTH:
        return False, (
            f"Your study material is too short (minimum {MIN_QUIZ_MATERIAL_LENGTH} "
            "characters). Please provide more content to quiz on."
        )

    if len(material) > MAX_QUIZ_MATERIAL_LENGTH:
        return False, (
            f"Your study material is too long (maximum {MAX_QUIZ_MATERIAL_LENGTH} "
            "characters). Please shorten it or split it into sections."
        )

    if difficulty not in VALID_QUIZ_DIFFICULTIES:
        return False, "Please select a valid quiz difficulty: Easy, Medium, or Hard."

    return True, ""


def validate_answer_improvement(question: str, answer: str) -> Tuple[bool, str]:
    """Validate the question/answer pair submitted to the Answer Improver."""
    question = question.strip()
    answer = answer.strip()

    if not question:
        return False, "Please enter the question you were answering."

    if len(question) < MIN_QUESTION_LENGTH:
        return False, "Please enter a more complete question."

    if len(question) > MAX_QUESTION_LENGTH:
        return False, (
            f"Your question is too long (maximum {MAX_QUESTION_LENGTH} characters)."
        )

    if not answer:
        return False, "Please enter your answer so it can be improved."

    if len(answer) < MIN_ANSWER_LENGTH:
        return False, "Please enter a more complete answer."

    if len(answer) > MAX_ANSWER_LENGTH:
        return False, (
            f"Your answer is too long (maximum {MAX_ANSWER_LENGTH} characters)."
        )

    return True, ""
