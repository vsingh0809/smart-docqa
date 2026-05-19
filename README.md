Smart Document Q&A with Evaluation Dashboard

A production-quality RAG (Retrieval-Augmented Generation) system with hybrid search, HyDE query expansion, and a live RAGAS evaluation dashboard — built to demonstrate real-world AI engineering practices.

Table of Contents

Overview
Live Demo
Architecture
RAGAS Evaluation Results
Key Technical Decisions
Tech Stack
Project Structure
Getting Started
Usage
Running Evaluation
Running Tests
Environment Variables
Contributing


Overview
Most RAG demos stop at "upload a PDF and ask questions." This project goes further:

Hybrid retrieval — combines dense vector search (ChromaDB) with sparse BM25 for keyword precision, fused via Reciprocal Rank Fusion (RRF)
HyDE (Hypothetical Document Embeddings) — LLM generates a hypothetical answer first, embeds it, and retrieves against that — dramatically improves results on short or vague queries
Contextual compression — retrieved chunks are filtered by an LLM before being passed to the answer chain, eliminating irrelevant noise
RAGAS evaluation — every answer can be scored on faithfulness, answer relevancy, context precision, and context recall — not just vibes


Live Demo
<!-- Replace with your Streamlit Cloud URL after deployment -->

▶ Live App on Streamlit Cloud

Show Image

Architecture
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                             │
│  Document (PDF/TXT/DOCX/URL)                                │
│       │                                                     │
│       ▼                                                     │
│  Recursive Text Splitter  (chunk=512, overlap=64)           │
│       │                                                     │
│       ▼                                                     │
│  OpenAI Embeddings  (text-embedding-3-small)                │
│       │                                                     │
│       ▼                                                     │
│  ChromaDB  (persisted local vector store)                   │
│  + BM25 Index  (in-memory sparse index)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                             │
│  User Query                                                 │
│       │                                                     │
│       ▼                                                     │
│  HyDE Expansion  →  Hypothetical doc embedding              │
│       │                                                     │
│       ▼                                                     │
│  Hybrid Retriever  (dense + BM25 → RRF fusion)              │
│       │                                                     │
│       ▼                                                     │
│  Contextual Compression  (LLMChainExtractor)                │
│       │                                                     │
│       ▼                                                     │
│  Answer Generation  (GPT-4o-mini + source citations)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                          │
│                                                             │
│  faithfulness · answer_relevancy · context_precision        │
│  context_recall  →  RAGAS  →  Streamlit Dashboard           │
└─────────────────────────────────────────────────────────────┘

RAGAS Evaluation Results
Tested on a 45-page technical PDF (LangChain documentation), 20 Q&A pairs with ground truth:
MetricScoreWhat It MeasuresFaithfulness0.91Every claim in the answer is grounded in the retrieved contextAnswer Relevancy0.87The answer actually addresses the question askedContext Precision0.83Retrieved chunks are relevant — not noisyContext Recall0.79All context needed to answer was retrieved
Hybrid search vs pure dense retrieval on the same dataset:
Retrieval MethodContext PrecisionContext RecallDense only0.710.68Hybrid (dense + BM25)0.830.79
Hybrid wins consistently on technical queries with exact terminology (function names, config keys, version numbers) — BM25 catches what dense embeddings miss.

Key Technical Decisions
Why hybrid search over pure dense retrieval?
Dense embeddings capture semantic meaning but miss exact keyword matches. A query like "what is the chunk_overlap default?" retrieves semantically similar text about chunking in general — not the specific config key. BM25 nails it. Combining both via Reciprocal Rank Fusion gives you the best of both worlds without needing a reranker model.
Why HyDE for query expansion?
Short queries like "refund policy?" have a weak embedding signal. HyDE instructs the LLM to generate a 2-3 sentence hypothetical answer, embeds that, and retrieves against it. The hypothetical document lives in the same embedding space as actual document chunks — so retrieval is far more targeted. The overhead is one extra LLM call (~0.001 USD with GPT-4o-mini).
Why contextual compression after retrieval?
Even good retrievers return chunks with surrounding noise. Contextual compression runs an LLMChainExtractor that filters each chunk to only the sentences relevant to the query — before they reach the answer chain. This reduces hallucination risk and keeps the context window lean.
Chunking strategy
RecursiveCharacterTextSplitter with chunk_size=512, chunk_overlap=64. The recursive splitter respects paragraph boundaries before falling back to sentence, then word splits. Overlap at 64 (12.5%) preserves cross-boundary context without doubling retrieval cost. Smaller chunks (256) hurt recall; larger (1024) hurt precision.

Tech Stack
ComponentTechnologyLLMGPT-4o-mini (OpenAI)Embeddingstext-embedding-3-small (OpenAI)Vector DBChromaDB (local, persisted)Sparse retrievalrank-bm25FrameworkLangChain LCELEvaluationRAGASFrontendStreamlitConfigpydantic-settingsTestingpytest + pytest-asyncioPython3.11+

Project Structure
smart-docqa/
├── .env.example                 # environment variable template
├── .gitignore
├── README.md
├── pyproject.toml
│
├── src/
│   └── smart_docqa/
│       ├── config.py            # pydantic-settings BaseSettings
│       ├── ingestion/
│       │   ├── loader.py        # PDF, TXT, DOCX, URL loaders
│       │   ├── chunker.py       # recursive splitter with metadata
│       │   └── embedder.py      # ChromaDB build + load
│       ├── retrieval/
│       │   ├── hybrid.py        # BM25 + dense RRF fusion
│       │   ├── hyde.py          # HyDE query expansion
│       │   └── compressor.py    # contextual compression
│       ├── chain/
│       │   ├── prompts.py       # system + QA prompt templates
│       │   └── qa_chain.py      # LCEL chain assembly
│       ├── evaluation/
│       │   ├── dataset.py       # EvalSample dataclass, JSON I/O
│       │   └── runner.py        # RAGAS runner → metric scores
│       └── utils/
│           └── logger.py        # structured logging
│
├── app/
│   ├── main.py                  # Streamlit entry point
│   └── pages/
│       ├── 1_chat.py            # file upload + streaming Q&A
│       └── 2_evaluation.py      # RAGAS dashboard + score history
│
└── tests/
    ├── conftest.py
    ├── test_loader.py
    ├── test_chunker.py
    ├── test_hybrid_retriever.py
    └── test_qa_chain.py

Getting Started
Prerequisites

Python 3.11+
An OpenAI API key

Installation
bash# 1. Clone the repository
git clone https://github.com/vsingh0809/smart-docqa.git
cd smart-docqa

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Set up environment variables
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
.env.example
envOPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=documents
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RETRIEVER_K=6
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

Never commit your .env file. It is listed in .gitignore by default.


Usage
Start the Streamlit app
bashstreamlit run app/main.py
Then open http://localhost:8501 in your browser.
Chat page

Upload a PDF, TXT, or DOCX file using the file uploader
Wait for the ingestion spinner to complete (chunks are cached for the session)
Ask questions in the chat input — answers stream token-by-token with source citations

Python API (programmatic use)
pythonfrom smart_docqa.ingestion.loader import load_document
from smart_docqa.ingestion.chunker import chunk_documents
from smart_docqa.ingestion.embedder import build_vectorstore
from smart_docqa.retrieval.hybrid import build_hybrid_retriever
from smart_docqa.retrieval.hyde import HyDERetriever
from smart_docqa.chain.qa_chain import build_qa_chain

docs = load_document("path/to/your/document.pdf")
chunks = chunk_documents(docs)
vs = build_vectorstore(chunks)
hybrid = build_hybrid_retriever(vs, chunks)
retriever = HyDERetriever(base_retriever=hybrid)
chain, _ = build_qa_chain(retriever)

answer = chain.invoke("What is the refund policy?")
print(answer)

Running Evaluation
Navigate to the Evaluation Dashboard page in the Streamlit app:

Upload and ingest a document on the Chat page first
Switch to the Evaluation page
Enter 3–20 question + ground truth pairs
Click Run RAGAS evaluation
Scores appear immediately and are saved to eval_results.json for history

All four RAGAS metrics are computed:
MetricWhat a low score meansfaithfulnessThe answer contains claims not backed by the retrieved context (hallucination)answer_relevancyThe answer is off-topic or doesn't address the questioncontext_precisionIrrelevant chunks are being retrieved (noisy retriever)context_recallKey information wasn't retrieved (coverage gap)

Running Tests
bash# run all tests
pytest

# with coverage report
pytest --cov=src/smart_docqa --cov-report=term-missing

# run a specific module
pytest tests/test_hybrid_retriever.py -v

Environment Variables
VariableDefaultDescriptionOPENAI_API_KEYrequiredYour OpenAI API keyCHROMA_PERSIST_DIR./chroma_dbDirectory for ChromaDB persistenceCOLLECTION_NAMEdocumentsChromaDB collection nameCHUNK_SIZE512Characters per chunkCHUNK_OVERLAP64Overlap between consecutive chunksRETRIEVER_K6Number of chunks to retrieveEMBEDDING_MODELtext-embedding-3-smallOpenAI embedding modelLLM_MODELgpt-4o-miniOpenAI chat model

Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Fork the repo
Create a feature branch (git checkout -b feat/your-feature)
Commit your changes using conventional commits (git commit -m "feat: add reranker support")
Push to the branch (git push origin feat/your-feature)
Open a pull request


License
MIT