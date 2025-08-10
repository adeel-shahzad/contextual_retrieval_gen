from pydantic import BaseModel
from typing import List

# --------- Data Models ---------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False