from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from smart_docqa.config import settings
from smart_docqa.retriever.compressor import doc_compressor
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from smart_docqa.chain.prompts import QA_PROMPT
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs:Document)->str:
    return "\n\n---\n\n".join(
        f"[source:{doc.metadata.get('source','unknown')}]\n {doc.page_content}"
        for doc in docs
    )

def build_qa_chain(retriever:BaseRetriever):
    llm=ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.open_api_key,
        temperature=0
    )

    compressor=doc_compressor(base_retriever=BaseRetriever,llm=llm)

    chain=(
        {
            "context":compressor|RunnableLambda(format_docs),
            "questions":RunnablePassthrough()
        }
        |QA_PROMPT
        |llm
        |StrOutputParser()
    )

    return chain,compressor


