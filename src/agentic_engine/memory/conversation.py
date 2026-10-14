import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TurnRecord(BaseModel):
    turn_id: int
    user_input: str
    agent_output: str
    timestamp: float = Field(default_factory=time.time)
    intent: Optional[str] = None
    tokens_used: int = 0

class ConversationMemoryManager:
    """
    Dynamic context window and conversation history management.
    Features sliding window buffer, token budget estimation, and session state tracking.
    """
    def __init__(self, max_turns: int = 10, max_token_budget: int = 4000):
        self.max_turns = max_turns
        self.max_budget = max_token_budget
        self.history: List[TurnRecord] = []
        self._current_turn = 0

    def add_turn(self, user_text: str, agent_text: str, intent: str = None) -> TurnRecord:
        self._current_turn += 1
        approx_tokens = (len(user_text.split()) + len(agent_text.split())) * 2
        record = TurnRecord(
            turn_id=self._current_turn,
            user_input=user_text,
            agent_output=agent_text,
            intent=intent,
            tokens_used=approx_tokens
        )
        self.history.append(record)
        # Apply sliding window
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        return record

    def get_context_for_prompt(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history:
            lines.append(f"User: {turn.user_input}")
            lines.append(f"Assistant: {turn.agent_output}")
        return "
".join(lines)

    def clear(self):
        self.history.clear()
        self._current_turn = 0
