from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env",extra="ignore")

    open_api_key:str
    chroma_persist_dir:str="./chroma_db"
    collection_name:str="documents"
    chunk_size:int=512
    chunk_overlap:int=64
    retriever_k:int=6
    embedding_model:str="text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

settings = Settings()