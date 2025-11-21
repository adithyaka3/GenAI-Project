from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class NotebookBlock(BaseModel):
    cell_type: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CodeBlock(NotebookBlock):
    cell_type: str = "code"
    outputs: List["OutputBlock"] = Field(default_factory=list)
    summary: Optional[str] = None

class MarkdownBlock(NotebookBlock):
    cell_type: str = "markdown"

class OutputBlock(BaseModel):
    output_type: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
