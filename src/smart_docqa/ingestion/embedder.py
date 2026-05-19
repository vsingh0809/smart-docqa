import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import SecretStr
from smart_docqa.config import settings

logger = logging.getLogger(__name__)


def get_embedding() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,  # wrap in SecretStr
    )

def doc_embedd(chunks: list[Document]) -> Chroma:
    if not chunks:
        raise ValueError("Cannot build vectorstore from empty chunk list.")

    logger.info(
        "Embedding %d chunks into collection '%s'",
        len(chunks),
        settings.collection_name,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding(),
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
    )

    logger.info("Vectorstore built and persisted to %s", settings.chroma_persist_dir)
    return vector_store


def load_chroma() -> Chroma:
    return Chroma(
        embedding_function=get_embedding(),   # call the function, not pass it
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )
