import logging
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset
from smart_docqa.evaluation.dataset import EvalSample
from smart_docqa.config import settings

logger = logging.getLogger(__name__)


def run_ragas(chain, retriever, samples: list[EvalSample]) -> dict:
    if not samples:
        raise ValueError("Provide at least one evaluation sample.")

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for sample in samples:
        try:
            answer = chain.invoke(sample.question)
            docs = retriever.get_relevant_documents(sample.question)
            contexts = [doc.page_content for doc in docs]
            rows["question"].append(sample.question)
            rows["answer"].append(answer)
            rows["contexts"].append(contexts)
            rows["ground_truth"].append(sample.ground_truth)
        except Exception:
            logger.exception("Failed to process sample: %s", sample.question)

    dataset = Dataset.from_dict(rows)

    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ))

    embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    ))

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    results = evaluate(dataset, metrics=metrics)
    result=results.to_pandas()

    scores = {
        "faithfulness": round(float(result["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(result["answer_relevancy"].mean()), 3),
        "context_precision": round(float(result["context_precision"].mean()), 3),
        "context_recall": round(float(result["context_recall"].mean()), 3),
    }

    logger.info("RAGAS scores: %s", scores)
    return scores