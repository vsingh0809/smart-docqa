from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from smart_docqa.config import settings
from smart_docqa.retrieval.compressor import doc_compressor
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from smart_docqa.chain.prompts import QA_PROMPT
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs:Document)->str:
    return "\n\n---\n\n".join(
        f"[source:{doc.metadata.get('source','unknown')}]\n {doc.page_content}"
        for doc in docs
    )

def build_qa_chain(retriever:BaseRetriever):
    llm=ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0
    )

    compressor=doc_compressor(base_retriever=retriever,llm=llm)

    chain=(
        {
            "context":compressor|RunnableLambda(format_docs),
            "question":RunnablePassthrough()
        }
        |QA_PROMPT
        |llm
        |StrOutputParser()
    )

    return chain,compressor


