import streamlit as st
from pathlib import Path
import tempfile
from smart_docqa.ingestion.loader import load_documents
from smart_docqa.ingestion.chunker import load_chunks
from smart_docqa.ingestion.embedder import doc_embedd
from smart_docqa.retrieval.hybrid import build_hybrid_retriever
from smart_docqa.retrieval.hyde import HydeRetriever
from smart_docqa.chain.qa_chain import build_qa_chain

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Document Q&A")

@st.cache_resource(show_spinner="Ingesting document...")
def ingest(file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    docs = load_documents(tmp_path)
    chunks = load_chunks(docs)
    vs = doc_embedd(chunks)
    hybrid = build_hybrid_retriever(vs, chunks)
    retriever = HydeRetriever(base_retriever=hybrid)
    chain, compressed = build_qa_chain(retriever)
    st.session_state.chain = chain          # ← set inside cache
    st.session_state.retriever = compressed
    return chain, compressed, len(chunks)

uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])

if uploaded:
    chain, retriever, n_chunks = ingest(uploaded.read(), uploaded.name)
    st.session_state.chain = chain          # ← set outside cache too
    st.session_state.retriever = retriever
    st.caption(f"{n_chunks} chunks indexed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask a question about your document"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            response = st.write_stream(chain.stream(question))

        st.session_state.messages.append({"role": "assistant", "content": response})