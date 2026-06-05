import logging
import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from smart_docqa.config import settings

logger = logging.getLogger(__name__)

@st.cache_resource
def get_dense_embedding_model():
    """Lazily loads the FastEmbed model into RAM once to prevent Azure OOM crashes."""
    logger.info("Initializing Dense Model into RAM...")
    return FastEmbedEmbeddings(model_name=settings.dense_embedding_model)

@st.cache_resource
def get_sparse_embedding_model():
    """Lazily loads the SPLADE sparse model into RAM once."""
    from langchain_qdrant import FastEmbedSparse
    logger.info("Initializing Sparse Model into RAM...")
    return FastEmbedSparse(model_name=settings.sparse_embedding_model)