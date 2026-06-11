import os
import logging
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from smart_docqa.config import settings
from smart_docqa.model_manager.model_manager import get_dense_embedding_model, get_sparse_embedding_model

logger = logging.getLogger(__name__)

def doc_embedd(chunks: list[Document]) -> QdrantVectorStore:
    if not chunks:
        raise ValueError("Cannot build vectorstore from empty chunk list.")

    logger.info("Embedding %d chunks to Qdrant Cloud...", len(chunks))

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_dense_embedding_model(),
        sparse_embedding=get_sparse_embedding_model(),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        force_recreate=True ,
        batch_size=50,  
        timeout=60, 
    )
    return vector_store