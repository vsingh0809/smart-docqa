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
    embedding_model: str = "models/gemini-embedding-2-preview"
    llm_model: str = "gemini-3.1-flash-lite"

settings = Settings()
