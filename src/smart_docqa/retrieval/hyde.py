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
    ("system", "Write a short factual passage (2-3 sentences) that answers the question."),
    ("human", "{question}"),
])

class HydeRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: ChatGoogleGenerativeAI = None # Define it at the class level

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize exactly once when the class is built
        self.llm = ChatGoogleGenerativeAI(     
            model=settings.llm_model,
            temperature=0,
            google_api_key=settings.google_api_key,
        )

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
        hypothetical_doc = hyde_chain.invoke({"question": query})
        return self.base_retriever.invoke(hypothetical_doc)