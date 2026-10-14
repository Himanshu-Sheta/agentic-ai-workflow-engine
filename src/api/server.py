from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.core.orchestrator import AgenticWorkflowOrchestrator

app = FastAPI(
    title="Agentic AI Workflow Engine",
    description="Enterprise Multi-Agent Orchestration, Hybrid RAG, and Safety Guardrails API",
    version="1.0.0"
)

# Seed Knowledge Base
SAMPLE_KB = [
    {
        "chunk_id": "eng_architecture_01",
        "text": "The Agentic AI Workflow Engine uses an Evaluator-Optimizer pattern with Reciprocal Rank Fusion hybrid retrieval.",
        "category": "architecture"
    },
    {
        "chunk_id": "eng_guardrails_02",
        "text": "All incoming user inputs undergo deterministic prompt injection scanning and PII masking prior to vector indexing.",
        "category": "security"
    },
    {
        "chunk_id": "eng_metrics_03",
        "text": "The RAG Triad benchmarks faithfulness, answer relevance, and context precision with automated ground-truth scoring.",
        "category": "evaluation"
    }
]

orchestrator = AgenticWorkflowOrchestrator(knowledge_base=SAMPLE_KB)

class WorkflowRequest(BaseModel):
    task_id: str = "task_001"
    query: str

class WorkflowResponse(BaseModel):
    task_id: str
    status: str
    route_selected: Optional[str]
    retrieved_chunks: int
    final_output: str
    hallucination_score: float
    latency_ms: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "agentic-ai-workflow-engine", "version": "1.0.0"}

@app.post("/v1/agent/execute", response_model=WorkflowResponse)
def execute_agent_task(req: WorkflowRequest):
    state = orchestrator.execute_workflow(req.task_id, req.query)
    return WorkflowResponse(
        task_id=state.task_id,
        status=state.status.value,
        route_selected=state.route_selected,
        retrieved_chunks=len(state.retrieved_contexts),
        final_output=state.final_output or "",
        hallucination_score=state.hallucination_score,
        latency_ms=state.latency_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
