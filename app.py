"""
app.py
------
Streamlit front-end for the AI Resume & Interview Assistant.

Features:
  1. Upload a resume (PDF or TXT) or use the bundled sample resume.
  2. Ask free-form questions about the resume ("What projects has this
     candidate worked on?").
  3. Generate tailored interview questions (optionally for a specific role).
  4. Get resume feedback / improvement suggestions.

Run with:  streamlit run app.py
"""

import os
import streamlit as st

from rag_agent import ResumeRAGAgent, load_text_from_bytes, load_text_from_file

SAMPLE_RESUME_PATH = os.path.join(os.path.dirname(__file__), "resume_info.txt")

st.set_page_config(page_title="AI Resume & Interview Assistant", page_icon="📄", layout="wide")

st.title("📄 AI Resume & Interview Assistant")
st.caption("Upload a resume to ask questions, generate interview questions, and get feedback.")

# --------------------------------------------------------------------------
# Sidebar: resume input
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Resume Source")
    uploaded_file = st.file_uploader("Upload a resume (.pdf or .txt)", type=["pdf", "txt"])
    use_sample = st.checkbox("Use bundled sample resume instead", value=uploaded_file is None)

    st.divider()
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if api_key_present:
        st.success("ANTHROPIC_API_KEY detected — using Claude for generation.")
    else:
        st.info(
            "No ANTHROPIC_API_KEY set. The app will still work using a "
            "built-in retrieval-based fallback (no external calls)."
        )

# --------------------------------------------------------------------------
# Load resume text + build the RAG agent (cached across reruns)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Indexing resume...")
def build_agent(resume_text: str) -> ResumeRAGAgent:
    return ResumeRAGAgent(resume_text)


resume_text = None
source_label = None

if uploaded_file is not None and not use_sample:
    resume_text = load_text_from_bytes(uploaded_file.read(), uploaded_file.name)
    source_label = uploaded_file.name
elif use_sample or uploaded_file is None:
    resume_text = load_text_from_file(SAMPLE_RESUME_PATH)
    source_label = "resume_info.txt (sample)"

if not resume_text or not resume_text.strip():
    st.warning("Couldn't extract any text from the resume. Please try another file.")
    st.stop()

agent = build_agent(resume_text)
st.caption(f"Currently loaded resume: **{source_label}**  •  {len(agent.chunks)} indexed sections")

with st.expander("View extracted resume text"):
    st.text(resume_text)

st.divider()

# --------------------------------------------------------------------------
# Tabs: Q&A / Interview Questions / Feedback
# --------------------------------------------------------------------------

tab_qa, tab_questions, tab_feedback = st.tabs(
    ["💬 Ask about the resume", "🎤 Generate interview questions", "📝 Get resume feedback"]
)

with tab_qa:
    st.subheader("Ask a question about this resume")
    question = st.text_input(
        "e.g. What backend frameworks has this candidate used?",
        key="qa_input",
    )
    if st.button("Get answer", key="qa_button") and question.strip():
        with st.spinner("Retrieving relevant sections and answering..."):
            answer = agent.answer_question(question)
        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander("Show retrieved resume sections used"):
            for chunk, score in agent.retrieve(question, top_k=3):
                st.markdown(f"- *(relevance {score:.2f})* {chunk}")

with tab_questions:
    st.subheader("Generate tailored interview questions")
    col1, col2 = st.columns([2, 1])
    with col1:
        role = st.text_input("Target role (optional)", placeholder="e.g. Backend Java Developer")
    with col2:
        n_questions = st.slider("Number of questions", min_value=3, max_value=10, value=5)

    if st.button("Generate questions"):
        with st.spinner("Generating interview questions..."):
            questions = agent.generate_interview_questions(role=role, n=n_questions)
        st.markdown("**Suggested interview questions:**")
        for i, q in enumerate(questions, start=1):
            st.markdown(f"{i}. {q}")

with tab_feedback:
    st.subheader("Resume feedback")
    st.write("Get suggestions on strengths, gaps, and improvements for this resume.")
    if st.button("Analyze resume"):
        with st.spinner("Analyzing resume..."):
            feedback = agent.generate_feedback()
        st.markdown("**Feedback:**")
        st.write(feedback)

st.divider()
st.caption(
    "This assistant uses TF-IDF retrieval over the resume text, with optional Claude "
    "generation when ANTHROPIC_API_KEY is set. No resume data is stored between sessions."
)
