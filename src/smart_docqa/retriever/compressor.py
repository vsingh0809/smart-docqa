import logging
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from smart_docqa.config import settings


logger=logging.getLogger(__name__)

def doc_compressor(base_retriever:BaseRetriever,llm:ChatOpenAI)->ContextualCompressionRetriever:
    
    compressor=LLMChainExtractor.from_llm(llm)

    compressed_Retriever=ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    ) 

    logger.debug(
        "ContextualCompressionRetriever built with model=%s on top of %s",
        settings.llm_model,
        type(base_retriever).__name__,
    )

    return compressed_Retriever
