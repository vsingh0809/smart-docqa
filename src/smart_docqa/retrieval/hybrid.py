import logging
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

def build_hybrid_retriever(vector_store: QdrantVectorStore, k: int = 4) -> BaseRetriever:
    """Builds a native hybrid retriever using Qdrant's internal database fusion."""
    logger.info("Building native Qdrant Hybrid Retriever")
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
        retrieval_mode=RetrievalMode.HYBRID
    )