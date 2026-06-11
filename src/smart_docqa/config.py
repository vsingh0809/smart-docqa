from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=str(ENV_FILE),extra="ignore")

    open_api_key:str | None = None
    google_api_key:str | None = None
    user_agent: str = "smart-docqa/0.1.0"
    chroma_persist_dir:str="./chroma_db"
    collection_name:str="documents"
    chunk_size:int=512
    chunk_overlap:int=64
    retriever_k:int=6
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "SmartQA"

    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "prithivida/Splade_PP_en_v1"
    llm_model: str = "gemini-3.1-flash-lite"

settings = Settings()
