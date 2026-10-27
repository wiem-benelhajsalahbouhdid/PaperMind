"""Web interface for the RAG project.

Run:  streamlit run app.py
"""
import html
import re
from collections import Counter

import streamlit as st

from rag import answer, chunks

# ---------- Page settings (must be the first Streamlit command) ----------
st.set_page_config(page_title="PaperMind", page_icon="📚", layout="centered")

# ---------- Custom style (CSS) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }

    .hero { padding: 1.6rem 0 0.4rem 0; }
    .hero h1 {
        font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.6rem;
        color: #1F2A44; margin: 0; letter-spacing: -0.02em;
    }
    .hero p { color: #6B6459; font-size: 1.02rem; margin-top: 0.4rem; }
    .accent { color: #C2410C; }

    .cite {
        display: inline-block; background: #FDE7D6; color: #9A3412;
        font-size: 0.78rem; font-weight: 600; padding: 0 0.42rem;
        border-radius: 6px; margin: 0 1px;
    }

    .source-card {
        background: #FFFFFF; border: 1px solid #E8E2D6; border-left: 4px solid #C2410C;
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.7rem;
    }
    .source-head { display: flex; justify-content: space-between; align-items: center; }
    .source-title { font-weight: 600; color: #1F2A44; font-size: 0.92rem; }
    .source-score { color: #6B6459; font-size: 0.8rem; }
    .bar { height: 5px; background: #F1ECE2; border-radius: 4px; margin: 0.45rem 0 0.55rem 0; }
    .bar > div { height: 5px; background: #C2410C; border-radius: 4px; }
    .source-text { color: #4A4540; font-size: 0.86rem; line-height: 1.5; }

    .paper-item { font-size: 0.85rem; color: #4A4540; padding: 0.15rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def highlight_citations(text):
    """Turn [1], [2]... in the answer into small coloured badges."""
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', text)


def show_sources(sources):
    """Show each retrieved chunk as a card with a similarity bar."""
    for n, s in enumerate(sources, start=1):
        width = max(0, min(100, int(s["score"] * 100)))
        preview = html.escape(s["text"][:420]) + "..."
        st.markdown(
            f"""
            <div class="source-card">
              <div class="source-head">
                <span class="source-title"><span class="cite">{n}</span> {html.escape(s['file'])} · page {s['page']}</span>
                <span class="source-score">similarity {s['score']:.2f}</span>
              </div>
              <div class="bar"><div style="width:{width}%"></div></div>
              <div class="source-text">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Memory of the conversation (kept between reruns) ----------
if "history" not in st.session_state:
    st.session_state.history = []      # list of {"question", "answer", "sources"}
if "pending" not in st.session_state:
    st.session_state.pending = None    # question clicked from the examples

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    k = st.slider("Number of sources (top-k)", min_value=1, max_value=8, value=4)
    show_src = st.toggle("Show sources", value=True)

    st.markdown("### 📄 Indexed papers")
    chunks_per_file = Counter(c["file"] for c in chunks)
    for name, count in sorted(chunks_per_file.items()):
        st.markdown(f'<div class="paper-item">• {html.escape(name)} <span style="color:#9C9488">({count} chunks)</span></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>Ask your <span class="accent">papers</span></h1>
      <p>Answers grounded in your research papers, with the exact page cited.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Example questions (only before the first question) ----------
examples = [
    "What is FedHealth and how does it protect privacy?",
    "Why does FedAvg often fail with medical data?",
    "What future directions are proposed for FL in healthcare?",
]
if not st.session_state.history:
    st.caption("Try one of these:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# ---------- Previous messages ----------
for turn in st.session_state.history:
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="📚"):
        st.markdown(highlight_citations(turn["answer"]), unsafe_allow_html=True)
        if show_src:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                show_sources(turn["sources"])

# ---------- New question ----------
question = st.chat_input("Ask a question about your papers...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Searching the papers and writing the answer..."):
            try:
                text, sources = answer(question, k)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.markdown(highlight_citations(text), unsafe_allow_html=True)
        if show_src:
            with st.expander(f"Sources ({len(sources)})", expanded=True):
                show_sources(sources)
    st.session_state.history.append({"question": question, "answer": text, "sources": sources})