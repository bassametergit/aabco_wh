"""Callback handlers used in the app."""
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler

    
class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass
    


