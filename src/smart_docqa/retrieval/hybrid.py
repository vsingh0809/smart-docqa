import logging
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from smart_docqa.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    vector_database: Chroma
    bm25: BM25Okapi
    corpus: list[Document]
    k: int = settings.retriever_k

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        dense_docs = self.vector_database.similarity_search(
            query=query,
            k=self.k * 2,
        )
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.k * 2]
        bm25_docs = [self.corpus[i] for i in bm25_indices]
        return self._reciprocal_rank_fusion(dense_docs, bm25_docs)

    def _reciprocal_rank_fusion(
        self,
        dense: list[Document],
        sparse: list[Document],
        k: int = 60,
    ) -> list[Document]:
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rank + k)  # ✅ brackets
            doc_map[key] = doc

        for rank, doc in enumerate(sparse):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rank + k)  # ✅ brackets
            doc_map[key] = doc

        fusion = sorted(scores, key=lambda x: scores[x], reverse=True)[: self.k]
        return [doc_map[key] for key in fusion]  # ✅ use fusion not scores


def build_hybrid_retriever(vector: Chroma, corpus: list[Document]) -> HybridRetriever:
    tokenized_corpus = [doc.page_content.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return HybridRetriever(vector_database=vector, bm25=bm25, corpus=corpus)