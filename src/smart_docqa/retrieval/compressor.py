import logging
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

def doc_compressor(base_retriever: BaseRetriever) -> ContextualCompressionRetriever:
    # Uses a tiny 90MB local model to filter bad chunks instantly
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )