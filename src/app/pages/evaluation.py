import streamlit as st
import json
from smart_docqa.evaluation.dataset import EvalSample, save_eval_results, load_eval_results

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")
st.title("RAGAS evaluation dashboard")

METRIC_DESCRIPTIONS = {
    "faithfulness": "Is the answer grounded in the retrieved context?",
    "answer_relevancy": "Does the answer address the question?",
    "context_precision": "Are the retrieved chunks actually relevant?",
    "context_recall": "Did we retrieve all chunks needed to answer?",
}

past = load_eval_results()
if past:
    latest = past[-1]
    cols = st.columns(4)
    for col, (metric, desc) in zip(cols, METRIC_DESCRIPTIONS.items()):
        score = latest.get(metric, 0)
        color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
        col.metric(label=metric.replace("_", " ").title(), value=f"{score:.3f}")

st.divider()
st.subheader("Run a new evaluation")

n = st.number_input("Number of test questions", min_value=1, max_value=20, value=3)
samples_input = []
for i in range(n):
    with st.expander(f"Sample {i+1}"):
        q = st.text_input("Question", key=f"q{i}")
        gt = st.text_area("Ground truth answer", key=f"gt{i}")
        if q and gt:
            samples_input.append(EvalSample(question=q, ground_truth=gt))

if st.button("Run RAGAS evaluation", disabled=len(samples_input) == 0):
    if "chain" not in st.session_state or "retriever" not in st.session_state:
        st.error("Upload and ingest a document on the Chat page first.")
    else:
        from smart_docqa.evaluation.runner import run_ragas
        with st.spinner("Evaluating..."):
            scores = run_ragas(
                st.session_state.chain,
                st.session_state.retriever,
                samples_input,
            )
        scores["timestamp"] = str(__import__("datetime").datetime.now())
        past.append(scores)
        save_eval_results(past)
        st.success("Done!")
        st.json(scores)