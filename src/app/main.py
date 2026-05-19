"""
app/main.py
-----------
Streamlit application entry point.

Run with:
    streamlit run app/main.py

This file does three things only:
  1. Configures logging before anything else imports.
  2. Sets global Streamlit page config (must be the very first st.* call).
  3. Renders the home / landing page with navigation hints.

The actual features live in app/pages/:
  - 1_chat.py       →  document upload + streaming Q&A
  - 2_evaluation.py →  RAGAS evaluation dashboard
"""
import os
os.environ.setdefault("USER_AGENT", "SmartDocQA/1.0")
import streamlit as st

#from smart_docqa.utils.logger import setup_logging

# ── Logging — must happen before any other import that uses logging ───────────
#setup_logging()

# ── Page config — must be the FIRST Streamlit call in the entire process ──────
st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Home page content ─────────────────────────────────────────────────────────
st.title("📄 Smart Document Q&A")
st.caption("Production-quality RAG · Hybrid Search · HyDE · RAGAS Evaluation")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💬 Chat with your document")
    st.markdown(
        """
        - Upload a **PDF, TXT, or DOCX** file
        - Ask questions in natural language
        - Answers stream token-by-token with **source citations**
        - Powered by **hybrid search** (dense + BM25) and **HyDE** query expansion
        """
    )
    if st.button("Open Chat →", use_container_width=True):
        st.switch_page("pages/chat.py")

with col2:
    st.subheader("📊 Evaluation Dashboard")
    st.markdown(
        """
        - Run **RAGAS evaluation** on your uploaded document
        - Measure **faithfulness**, **answer relevancy**,
          **context precision**, and **context recall**
        - Compare scores across runs with history tracking
        - Identify exactly where your RAG pipeline is failing
        """
    )
    if st.button("Open Evaluation →", use_container_width=True):
        st.switch_page("pages/evaluation.py")

st.markdown("---")

with st.expander("⚙️  How it works", expanded=False):
    st.markdown(
        """
        **Ingestion pipeline**
        1. Document loaded (PDF / TXT / DOCX / URL)
        2. `RecursiveCharacterTextSplitter` splits into 512-char chunks with 64-char overlap
        3. Chunks embedded with `text-embedding-3-small` and stored in **ChromaDB**
        4. A **BM25** sparse index is built in memory over the same chunks

        **Query pipeline**
        1. User query expanded via **HyDE** (LLM generates a hypothetical answer, embeds it)
        2. **Hybrid retriever** fuses dense ChromaDB results + BM25 results via RRF
        3. **Contextual compression** (LLMChainExtractor) strips irrelevant sentences
        4. GPT-4o-mini generates a grounded answer with source citations

        **Evaluation**
        - [RAGAS](https://docs.ragas.io) scores every answer against ground truth
        - Results persisted to `eval_results.json` for history comparison
        """
    )

with st.expander("🛠️  Tech stack", expanded=False):
    st.markdown(
        """
        | Component | Technology |
        |---|---|
        | LLM | GPT-4o-mini |
        | Embeddings | text-embedding-3-small |
        | Vector DB | ChromaDB (local) |
        | Sparse retrieval | rank-bm25 |
        | Framework | LangChain LCEL |
        | Evaluation | RAGAS |
        | Frontend | Streamlit |
        | Config | pydantic-settings |
        """
    )

st.sidebar.success("Select a page above to get started.")