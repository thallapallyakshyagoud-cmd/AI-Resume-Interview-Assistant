"""
rag_agent.py
------------
Core RAG (Retrieval-Augmented Generation) logic for the AI Resume &
Interview Assistant.

Responsibilities:
  1. Load/parse resume text (from .txt or .pdf).
  2. Chunk the resume into retrievable passages.
  3. Build a TF-IDF index over the chunks (no external vector DB needed).
  4. Retrieve the most relevant chunks for a given query.
  5. Generate answers, interview questions, and feedback using the
     Groq API if GROQ_API_KEY is configured, otherwise falling back
     to simple template-based generation so the app still works offline.
"""

import os
import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# --------------------------------------------------------------------------
# Text loading
# --------------------------------------------------------------------------

def load_text_from_file(file_path: str) -> str:
    """Load raw text from a .txt or .pdf resume file."""
    if file_path.lower().endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed; cannot read PDF files.")

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Load raw text from an uploaded file's bytes."""
    if filename.lower().endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed; cannot read PDF files.")

        import io

        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return file_bytes.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, max_words: int = 80) -> List[str]:
    """
    Split resume text into small chunks for retrieval.
    """

    raw_blocks = re.split(r"\n\s*\n", text.strip())

    chunks = []

    for block in raw_blocks:
        block = block.strip()

        if not block:
            continue

        words = block.split()

        if len(words) <= max_words:
            chunks.append(block)

        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))

    return chunks if chunks else [text.strip()]


# --------------------------------------------------------------------------
# RAG Agent
# --------------------------------------------------------------------------

class ResumeRAGAgent:
    """
    Lightweight RAG agent over a single resume.

    Retrieval:
        TF-IDF + cosine similarity

    Generation:
        Groq API when GROQ_API_KEY is configured

    Fallback:
        Local template-based responses when no API key is configured.
    """

    def __init__(self, resume_text: str):

        self.resume_text = resume_text

        # Create resume chunks
        self.chunks: List[str] = chunk_text(resume_text)

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # Build document matrix
        self.doc_matrix = self.vectorizer.fit_transform(self.chunks)

        # Groq client
        self.client = None

        if _GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
            self.client = Groq(
                api_key=os.environ.get("GROQ_API_KEY")
            )

    # ----------------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Return the top_k most relevant resume chunks.
        """

        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vec,
            self.doc_matrix
        ).flatten()

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]

    # ----------------------------------------------------------------------
    # Groq LLM
    # ----------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        """
        Call Groq if configured.
        Otherwise return None so the application can use fallback logic.
        """

        if not self.client:
            return None

        try:

            response = self.client.chat.completions.create(
                model=os.environ.get(
                    "LLM_MODEL",
                    "openai/gpt-oss-20b"
                ),

                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],

                max_tokens=600,
                temperature=0.2
            )

            return response.choices[0].message.content

        except Exception as e:

            return (
                "[LLM error, showing fallback response] "
                + str(e)
            )

    # ----------------------------------------------------------------------
    # Resume Question Answering
    # ----------------------------------------------------------------------

    def answer_question(self, question: str) -> str:
        """
        Answer a question about the resume using retrieved context.
        """

        top_chunks = self.retrieve(
            question,
            top_k=3
        )

        context = (
            "\n\n".join(
                chunk for chunk, _ in top_chunks
            )
            or self.resume_text[:1000]
        )

        llm_answer = self._call_llm(

            system=(
                "You are an assistant that answers questions about "
                "a candidate's resume using ONLY the provided resume "
                "context. Be concise and factual. Do not invent "
                "information that is not present in the resume."
            ),

            user=(
                f"Resume context:\n{context}\n\n"
                f"Question: {question}"
            )
        )

        if llm_answer:
            return llm_answer

        # Offline fallback
        if top_chunks:

            return (
                "Based on the resume:\n\n"
                + top_chunks[0][0]
            )

        return (
            "I couldn't find anything relevant "
            "to that question in the resume."
        )

    # ----------------------------------------------------------------------
    # Interview Questions
    # ----------------------------------------------------------------------

    def generate_interview_questions(
        self,
        role: str = "",
        n: int = 5
    ) -> List[str]:
        """
        Generate interview questions based on the resume.
        """

        context = self.resume_text[:2500]

        role_line = (
            f" for a {role} position"
            if role
            else ""
        )

        llm_out = self._call_llm(

            system=(
                "You are a technical interviewer. Generate sharp, "
                "specific interview questions based on the candidate's "
                "actual resume content. Reference real projects, "
                "skills, or experience listed in the resume. "
                "Return each question on its own line, numbered."
            ),

            user=(
                f"Resume:\n{context}\n\n"
                f"Generate {n} interview questions"
                f"{role_line} based on this resume."
            )
        )

        if llm_out:

            questions = [
                q.strip(" -0123456789.")
                for q in llm_out.split("\n")
                if q.strip()
            ]

            return (
                questions[:n]
                if questions
                else [llm_out]
            )

        # Offline fallback
        skills = self._extract_keywords()

        templates = [

            "Can you walk me through the '{s}' project mentioned in your resume?",

            "How have you used {s} in a real project?",

            "What was the most challenging part of your experience with {s}?",

            "Tell me about a time you improved performance or quality using {s}.",

            "How would you explain {s} to someone unfamiliar with it?"
        ]

        questions = []

        for i in range(n):

            skill = (
                skills[i % len(skills)]
                if skills
                else "your resume"
            )

            template = templates[i % len(templates)]

            questions.append(
                template.format(s=skill)
            )

        return questions

    # ----------------------------------------------------------------------
    # Resume Feedback
    # ----------------------------------------------------------------------

    def generate_feedback(self) -> str:
        """
        Generate resume feedback using Groq or fallback heuristics.
        """

        context = self.resume_text[:3000]

        llm_out = self._call_llm(

            system=(
                "You are a career coach reviewing a resume. "
                "Give concise, actionable feedback covering "
                "strengths, gaps, and 3-5 concrete improvement "
                "suggestions. Be specific to the content provided "
                "and avoid generic advice."
            ),

            user=(
                f"Resume:\n{context}\n\n"
                "Provide feedback on this resume."
            )
        )

        if llm_out:
            return llm_out

        # Offline fallback
        feedback = []

        text_lower = self.resume_text.lower()

        if "summary" not in text_lower:
            feedback.append(
                "- Consider adding a short professional summary at the top."
            )

        if not re.search(r"\d+%", self.resume_text):
            feedback.append(
                "- Add quantified achievements "
                "(e.g., 'reduced X by 30%') to strengthen impact."
            )

        if "project" not in text_lower:
            feedback.append(
                "- Include a Projects section highlighting hands-on work."
            )

        if "certification" not in text_lower:
            feedback.append(
                "- Listing relevant certifications can strengthen your profile."
            )

        if not feedback:
            feedback.append(
                "- Resume covers the key sections well; "
                "focus on tailoring keywords to each job description."
            )

        return (
            "Fallback feedback (no LLM configured):\n"
            + "\n".join(feedback)
        )

    # ----------------------------------------------------------------------
    # Internal utilities
    # ----------------------------------------------------------------------

    def _extract_keywords(
        self,
        top_n: int = 8
    ) -> List[str]:
        """
        Pull out top TF-IDF terms across the resume
        as rough skill keywords.
        """

        scores = np.asarray(
            self.doc_matrix.sum(axis=0)
        ).flatten()

        terms = np.array(
            self.vectorizer.get_feature_names_out()
        )

        top_idx = np.argsort(scores)[::-1][:top_n]

        return [
            terms[i]
            for i in top_idx
        ]