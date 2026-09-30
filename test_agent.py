"""
test_agent.py
-------------
Basic unit tests for rag_agent.py.

Run with:  python -m pytest test_agent.py -v
       or:  python test_agent.py
"""

import os
import unittest

from rag_agent import ResumeRAGAgent, chunk_text, load_text_from_file

SAMPLE_RESUME_PATH = os.path.join(os.path.dirname(__file__), "resume_info.txt")


class TestChunking(unittest.TestCase):

    def test_chunk_text_splits_on_blank_lines(self):
        text = "Section one line.\n\nSection two line.\n\nSection three line."
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 3)

    def test_chunk_text_handles_empty_string(self):
        chunks = chunk_text("")
        self.assertTrue(len(chunks) >= 1)

    def test_chunk_text_splits_long_blocks(self):
        long_block = " ".join(["word"] * 300)
        chunks = chunk_text(long_block, max_words=80)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.split()), 80)


class TestResumeRAGAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Force offline/fallback mode for deterministic tests, regardless
        # of whether ANTHROPIC_API_KEY happens to be set in the environment.
        cls._orig_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        resume_text = load_text_from_file(SAMPLE_RESUME_PATH)
        cls.agent = ResumeRAGAgent(resume_text)

    @classmethod
    def tearDownClass(cls):
        if cls._orig_key:
            os.environ["ANTHROPIC_API_KEY"] = cls._orig_key

    def test_agent_indexes_chunks(self):
        self.assertGreater(len(self.agent.chunks), 0)

    def test_retrieve_returns_relevant_chunk(self):
        results = self.agent.retrieve("Spring Boot REST APIs", top_k=3)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertIn("Spring Boot", top_chunk)
        self.assertGreater(score, 0)

    def test_retrieve_irrelevant_query_returns_low_or_no_results(self):
        results = self.agent.retrieve("astrophysics quantum gravity", top_k=3)
        # Either nothing relevant, or very low similarity scores
        for _, score in results:
            self.assertLess(score, 0.3)

    def test_answer_question_fallback_returns_text(self):
        answer = self.agent.answer_question("What frameworks does the candidate know?")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_generate_interview_questions_returns_requested_count(self):
        questions = self.agent.generate_interview_questions(role="Java Developer", n=5)
        self.assertEqual(len(questions), 5)
        for q in questions:
            self.assertIsInstance(q, str)
            self.assertGreater(len(q), 0)

    def test_generate_feedback_returns_text(self):
        feedback = self.agent.generate_feedback()
        self.assertIsInstance(feedback, str)
        self.assertGreater(len(feedback), 0)

    def test_extract_keywords_returns_terms(self):
        keywords = self.agent._extract_keywords(top_n=5)
        self.assertEqual(len(keywords), 5)


if __name__ == "__main__":
    unittest.main()
