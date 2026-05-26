# MSADS RAG Assistant

Production-style Retrieval-Augmented Generation (RAG) chatbot for the University of Chicago MS in Applied Data Science (MSADS) program website.

The system crawls MSADS program pages, builds a hybrid retrieval pipeline, reranks retrieved chunks, and generates grounded answers using GPT-based generation.

---

# Project Overview

This project demonstrates how modern enterprise-grade RAG systems are designed and evaluated.

The assistant supports:

- semantic search over MSADS website content
- grounded answer generation
- hybrid retrieval
- reranking
- semantic intent routing
- curriculum-aware chunking
- explainable retrieval visualization

The project was designed as a production-style AI assistant rather than a simple chatbot demo.

---

# Final Production Architecture (A2)

```text
User Query
    ↓
Semantic Intent Router
    ↓
Hybrid Retrieval
(BM25 + Dense Retrieval)
    ↓
Reciprocal Rank Fusion (RRF)
    ↓
Cross-Encoder Reranker
    ↓
Grounded LLM Generation
    ↓
Answer + Retrieved Sources
```

---

# Key Features

## Hybrid Retrieval

The system combines:

- Dense semantic retrieval
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)

This improves retrieval robustness for both:
- semantic questions
- keyword-heavy curriculum/admissions queries

---

## Cross-Encoder Reranking

Retrieved chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves:
- retrieval precision
- answer grounding
- context relevance

---

## Semantic Intent Routing

A lightweight semantic query router prevents unnecessary retrieval for:
- greetings
- small talk
- out-of-scope questions

Examples:

```text
"hello"
"how are you?"
"what's the weather?"
```

These queries bypass retrieval and receive direct responses.

---

## Curriculum-Aware Chunking

Special curriculum chunks were created for:

```text
course-progressions
```

This preserves:
- course names
- curriculum structure
- electives
- capstone information

Special chunk type:

```text
curriculum_catalog
```

This significantly improves retrieval for questions like:

```text
List names of all courses
What electives are available?
What are the core courses?
```

---

## Explainable RAG UI

The Streamlit UI includes:

- pipeline progress visualization
- intent routing display
- retrieval source inspection
- reranker scores
- chunk metadata
- suggested follow-up questions
- architecture sidebar
- evaluation metrics

---

# Technologies Used

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | GPT-4o-mini |
| Crawling | Requests + BeautifulSoup |
| Evaluation | Hit Rate + RAGAS |

---

# Repository Structure

```text
msads-rag-assistant-v1/
│
├── app.py
├── run_pipeline.py
├── config.yaml
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
│
├── outputs/
│
├── src/
│   ├── chunking/
│   ├── crawling/
│   ├── evaluation/
│   ├── generation/
│   ├── indexing/
│   ├── retrieval/
│   └── utils/
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repo-url>
cd msads-rag-assistant-v1
```

## 2. Create Virtual Environment

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create `.env`

```env
OPENAI_API_KEY=your_openai_api_key
```

Optional:

```env
HF_TOKEN=your_huggingface_token
```

---

# Running the Pipeline

Update `config.yaml`

```yaml
pipeline:
  run_crawling: true
  run_chunking: true
  run_indexing: true
  run_retrieval_eval: true
  run_generation_eval: false
```

Run:

```bash
python run_pipeline.py
```

---

# Running the Chatbot

```bash
streamlit run app.py
```

---

# Retrieval Evaluation

Metric used:

```text
Hit Rate@K
```

Final evaluation:

```text
Hit Rate@5 = 1.00
```

This means every evaluation query retrieved at least one relevant source within the top-5 retrieved results.

---

# Generation Evaluation

Optional RAGAS evaluation supports:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness

---

# Sample Questions

## Admissions

- What are the admission requirements?
- Are GRE scores required?
- What documents are needed for the application?

## Curriculum

- What are the core courses?
- List names of all courses offered in the program.
- What electives are available?
- What is included in the capstone project?

## International Students

- Does the program support CPT/OPT?
- Can international students apply?
- What English proficiency tests are accepted?

## Program Structure

- What is the difference between online and in-person programs?
- Is part-time enrollment available?
- How long does it take to complete the degree?

## Career Outcomes

- What career outcomes do graduates achieve?
- Does the program provide career support?

---

# Architectural Decisions

## Why Hybrid Retrieval?

Dense retrieval alone may miss:
- exact keywords
- course names
- admissions terminology

BM25 alone may miss:
- semantic meaning
- paraphrased questions

Hybrid retrieval combines both strengths.

## Why Reranking?

Initial retrieval often returns noisy chunks.

Cross-encoder reranking improves:
- precision
- grounding
- answer quality

## Why Curriculum-Aware Chunking?

Course progression pages are highly structured.

Generic recursive chunking split course lists incorrectly.

Custom curriculum chunks preserved:
- course names
- quarter structure
- curriculum semantics

## Why Semantic Intent Routing?

Avoids unnecessary retrieval for:
- greetings
- small talk
- unrelated queries

This improves:
- latency
- cost
- UX

---

# Future Improvements

Potential next steps:

- Parent-child retrieval
- OCR + image captioning
- Multimodal RAG
- Conversational memory
- Feedback analytics
- Citation highlighting
- Agentic workflows
- Evaluation dashboard
- LangGraph orchestration

---

# Demo Features

The chatbot UI demonstrates:

- explainable retrieval
- retrieval pipeline visualization
- source transparency
- confidence display
- hybrid search architecture
- grounded answer generation

---


