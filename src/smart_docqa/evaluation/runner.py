from ragas import evaluate
from ragas.metrics import(_faithfulness,_answer_relevance,_context_precision,_context_recall)
from smart_docqa.evaluation.dataset import EvalSample
import logging
from datasets import Dataset

METRICS = [_faithfulness, _answer_relevance, _context_precision, _context_recall]

logger=logging.getLogger(__name__)

def run_ragas(chain,retriever,samples:dict[EvalSample])->dict:
    if not samples:
     raise ValueError("Provide at least one evaluation sample.")
    
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for sample in samples:
       try:
          answer=chain.invoke(sample.question)
          docs=retriever.get_relevant_documents(sample.question)
          contexts=[doc.page_content  for doc in docs]
          rows["answer"]=answer
          rows["question"]=sample.question
          rows["contexts"]=contexts
          rows["ground_truth"]=sample.ground_truth
       except Exception:
          logger.exception("Failed to process sample: %s", sample.question)

    dataset=Dataset.from_dict(rows)
    result=evaluate(dataset=dataset,metrics=METRICS)   

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 3),
        "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        "context_precision": round(float(result["context_precision"]), 3),
        "context_recall": round(float(result["context_recall"]), 3),
    }

    logger.info("RAGAS scores: %s", scores)
    return scores     

