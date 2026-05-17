from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smart_docqa.config import settings

def load_chunk(docs:list[Document])->list[Document]:

    if not docs:
        raise FileExistsError("there is no such docs")
    
    doc_splitter=RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n","\n"," ","","."],
        add_start_index=True
    )

    chunks=doc_splitter.split_documents(docs)

    for i,chunk in enumerate(chunks):
        chunk.metadata["chunk_index"]=i
        chunk.metadata["chunk_count"]=len(chunk)

    return chunks



