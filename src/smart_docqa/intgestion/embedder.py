import logging
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from smart_docqa.config import settings
from langchain_core.documents import Document

logger=logging(__name__)

def get_embedding()->OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.llm_model,
        api_key=settings.open_api_key
    )

def doc_embedd(chunks:list[Document])-> Chroma:
    if not chunks:
        raise ValueError("Cannot build vectorstore from empty chunk list.")
    logger.info("Embedding %d chunks into collection '%s'", len(chunks), settings.collection_name)

    vector_store=Chroma.from_documents(
        embedding=get_embedding,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
        documents=chunks
    )

    logger.info("Vectorstore built and persisted to %s", settings.chroma_persist_dir)
    return vector_store

def load_Chroma()->Chroma:
    return Chroma(
        embedding_function=doc_embedd,
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name
    )
    

