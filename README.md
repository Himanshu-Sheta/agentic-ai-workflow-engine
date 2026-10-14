# 🧠 Agentic AI Workflow & Guardrailed RAG Orchestration Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_REST-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2_Contracts-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Sparse_Search-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-success)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An enterprise-grade, high-performance **Multi-Agent Orchestration Framework** in Python. Built for autonomous task routing, **Hybrid Information Retrieval** (BM25 sparse + Dense Vector embeddings fused with Reciprocal Rank Fusion), **Self-Reflective reasoning**, and **real-time safety and hallucination guardrails**.

---

## 🎯 Architectural Overview (Evaluator-Optimizer Workflow)

```text
                                [ User Prompt ]
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   1. AI Safety Shield             │
                     │   - Prompt Injection Detection    │
                     │   - Automated PII Redaction       │
                     └─────────────────┬─────────────────┘
                                       │ (Sanitized Prompt)
                                       ▼
                     ┌───────────────────────────────────┐
                     │   2. Dynamic Intent Router        │
                     │   - Knowledge / Code / Data / Gen │
                     └─────────────────┬─────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
                     ▼ (RAG Path)                        ▼ (Code/Data Path)
       ┌───────────────────────────┐       ┌───────────────────────────┐
       │ 3. Hybrid Retrieval       │       │ 4. Specialized Agents     │
       │ - Sparse BM25 (Lexical)   │       │ - Data Analysis Engine    │
       │ - Dense Vector Cosine     │       │ - Code Validation Loop    │
       │ - Reciprocal Rank Fusion  │       └─────────────┬─────────────┘
       └─────────────┬─────────────┘                     │
                     │ (Top Chunks)                      │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   5. Synthesis & Draft Reasoner   │
                     │   - Grounded Context Assembly     │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   6. Hallucination & Fact Checker │
                     │   - Sentence-level Entailment     │
                     │   - Self-Correction Iteration     │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     [ Verified, Fact-Checked Output ]
```

---

## 🚀 Key Modules & Engineering Highlights

* **Evaluator-Optimizer Multi-Agent Architecture**: Decouples query understanding, retrieval, and verification into modular, observable steps.
* **Hybrid Search with Reciprocal Rank Fusion (RRF)**: Combines keyword-level precision from the **BM25 Okapi** algorithm with semantic depth from **Dense Vector Embeddings**, mitigating vocabulary mismatch.
* **Defense-in-Depth Safety Shield**: Intercepts jailbreaks and prompt injection attempts, while anonymizing emails, phone numbers, credit cards, and German IBANs before processing.
* **Automated Groundedness & Hallucination Detector**: Scans each generated statement against context tokens to calculate an exact faithfulness score, blocking unsourced claims.
* **RAG Triad Evaluation Suite**: Built-in benchmark harness computing *Context Relevance*, *Faithfulness*, and *Answer Relevance*.

---

## 🛠️ Quickstart & Local Setup

### 1. Installation
```bash
git clone https://github.com/Himanshu-Sheta/agentic-ai-workflow-engine.git
cd agentic-ai-workflow-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the FastAPI Server
```bash
python -m src.api.server
```
Interactive OpenAPI documentation will be accessible at: `http://localhost:8000/docs`.

### 3. Query the Orchestration Engine
```bash
curl -X POST "http://localhost:8000/v1/agent/execute" \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "req_101",
       "query": "Explain the software architecture of hybrid RAG"
     }'
```

Sample Response:
```json
{
  "task_id": "req_101",
  "status": "COMPLETED",
  "route_selected": "KNOWLEDGE_RETRIEVAL (confidence: 0.92)",
  "retrieved_chunks": 1,
  "final_output": "Based on verified knowledge retrieval [eng_architecture_01]: The Agentic AI Workflow Engine uses an Evaluator-Optimizer pattern with Reciprocal Rank Fusion hybrid retrieval. This directly addresses your request regarding 'Explain the software architecture of hybrid RAG'.",
  "hallucination_score": 0.0,
  "latency_ms": 3.42
}
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## 👨‍💻 Author
**Himanshu Sheta**  
M.Sc. Applied Computer Science — Technische Hochschule Deggendorf  
Portfolio: [himanshusheta-portfolio.onrender.com](https://himanshusheta-portfolio.onrender.com)  
LinkedIn: [linkedin.com/in/himanshu-sheta](https://www.linkedin.com/in/himanshu-sheta)
