import logging
from pathlib import Path
from typing import Union
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader
)

logger=logging.getLogger(__name__)

LOADERS={
     ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}

def load_documents(source:Union[str,Path])->list[Document]:
    """Load a document from a file path or URL."""
    source= str(source)

    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Loading URL: %s", source)
        loader = WebBaseLoader(source)
        return loader.load()
    
    path=Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix= path.suffix.lower()
    loader_cls=LOADERS.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(LOADERS)}")
    
    logger.info("Loading %s file: %s", suffix, path)
    loader= loader_cls(str(path))
    docs= loader.load()

    for doc in docs:
        doc.metadata["source"]=str(path)
        doc.metadata["file_type"]=suffix

    return docs    

