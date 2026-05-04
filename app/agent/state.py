from typing import Optional, TypedDict, Annotated, List
import operator 
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    chunks: Annotated[list, operator.add]
    intent: str
    answer: str
    has_enough_context: bool
    error: Optional[str]
    retry_count: int
    query: str
    