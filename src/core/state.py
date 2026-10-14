from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AgentRole(str, Enum):
    ROUTER = "ROUTER"
    RETRIEVER = "RETRIEVER"
    REASONER = "REASONER"
    CRITIC = "CRITIC"
    GUARDRAIL = "GUARDRAIL"

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Message(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentExecutionState(BaseModel):
    task_id: str
    user_query: str
    status: TaskStatus = TaskStatus.PENDING
    route_selected: Optional[str] = None
    retrieved_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    draft_response: Optional[str] = None
    critique_notes: List[str] = Field(default_factory=list)
    final_output: Optional[str] = None
    tokens_consumed: int = 0
    latency_ms: float = 0.0
    guardrail_passed: bool = True
    hallucination_score: float = 0.0
