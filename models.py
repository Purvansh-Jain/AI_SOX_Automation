from dataclasses import dataclass
from langchain.agents import AgentState
from typing import Optional, Dict, List

@dataclass
class Message:
    content: str
    role: str = "assistant"

class AnalystState(AgentState):
    rcm_file_path: Optional[str] = None
    trail_balance_file_path: Optional[str] = None
    automation_excel_path: Optional[str] = None
    automation_pdf_path: Optional[str] = None
    rcm_column_mapping: Optional[Dict[str, List[str]]] = None
