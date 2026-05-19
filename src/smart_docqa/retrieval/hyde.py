import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_google_genai import ChatGoogleGenerativeAI
from smart_docqa.config import settings

logger = logging.getLogger(__name__)

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a document writing assistant. Write a short factual passage (2-3 sentences) that would answer the following question if it appeared in a document."),
    ("human", "{question}"),
])

class HydeRetriever(BaseRetriever):
    base_retriever: BaseRetriever

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(          # ✅ correct spelling
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        llm = ChatGoogleGenerativeAI(     # ✅ Gemini instead of OpenAI
            model=settings.llm_model,
            temperature=0,
            google_api_key=settings.google_api_key,
        )
        hyde_chain = HYDE_PROMPT | llm | StrOutputParser()   # ✅ added ()
        hypothetical_doc = hyde_chain.invoke({"question": query})
        return self.base_retriever.get_relevant_documents(hypothetical_doc)